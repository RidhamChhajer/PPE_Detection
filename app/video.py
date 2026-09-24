"""Complete timestamp-preserving, silent H.264 video processing."""
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path

import av
import cv2

from .detector import TensorRTEngine, model_specs


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: Fraction
    time_base: Fraction
    duration: Fraction
    timeline_hash: str
    codec: str
    audio_streams: int


def inspect_video(path, progress=None):
    """Decode every frame; never trust container frame-count metadata alone."""
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Select a video containing exactly one video stream')
        stream = container.streams.video[0]
        fps = stream.average_rate
        if fps is None or not 0 < fps <= 240:
            raise ValueError('The video must have a valid frame rate up to 240 FPS')
        width, height = stream.width, stream.height
        if width % 2 or height % 2 or min(width, height) < 16:
            raise ValueError('Browser H.264 output requires even dimensions of at least 16 pixels')
        if width * height > 3840 * 2160:
            raise ValueError('Maximum supported video size is 8.3 megapixels (4K)')
        first = previous = None
        duration = Fraction(0)
        digest = hashlib.sha256()
        count = 0
        for frame in container.decode(stream):
            if (frame.width, frame.height) != (width, height):
                raise ValueError('Videos that change resolution are not supported')
            if getattr(frame, 'rotation', 0):
                raise ValueError('Normalize video rotation metadata before uploading')
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Video contains a frame without a presentation timestamp')
            timestamp = frame.pts * frame.time_base
            if first is None:
                first = timestamp
            if previous is not None and timestamp <= previous:
                raise ValueError('Video timestamps must be strictly increasing')
            previous = timestamp
            relative = timestamp - first
            digest.update(f'{relative.numerator}/{relative.denominator};'.encode())
            frame_duration = frame.duration * frame.time_base if frame.duration else 1 / fps
            duration = relative + frame_duration
            count += 1
            if progress and count % 50 == 0:
                progress(count, stream.frames or count)
        if count == 0:
            raise ValueError('No decodable video frames were found')
        if stream.frames and count != stream.frames:
            raise ValueError(f'Incomplete video decode: {count}/{stream.frames} frames')
        return VideoInfo(width, height, count, fps, stream.time_base, duration,
                         digest.hexdigest(), stream.codec_context.name, len(container.streams.audio))


def draw_detections(frame, label, detections):
    for x1, y1, x2, y2, confidence in detections:
        left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        text = f'{label} {confidence:.2f}'
        scale = max(0.4, min(frame.shape[1] / 1400, 0.8))
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        text_left = max(0, min(left, frame.shape[1] - text_width - 4))
        text_y = max(text_height + 4, top - 5)
        cv2.rectangle(frame, (text_left, text_y-text_height-3),
                      (text_left+text_width+3, text_y+baseline), (20, 28, 20), -1)
        cv2.putText(frame, text, (text_left+1, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), 1, cv2.LINE_AA)
    return frame


def process_video(source, destination, models_directory, progress=None, engine_factory=TensorRTEngine):
    """Run every installed model on every original frame, then verify the output.

    engine_factory is injectable for tests; the application always uses TensorRT.
    The only output artifact is the complete annotated video, with no audio.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('Input and output must be different files')
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing output video')
    specifications = model_specs(models_directory)

    def update(stage, count, total):
        if progress:
            progress(stage, count, total)

    update('Inspecting input', 0, 1)
    original = inspect_video(source, lambda count, total: update('Inspecting input', count, total))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + '.partial.mp4')
    if temporary.exists():
        raise FileExistsError('A partial output already exists for this job')
    detections = 0
    try:
        with ExitStack() as resources:
            engines = [(spec, resources.enter_context(engine_factory(spec.path))) for spec in specifications]
            reader = resources.enter_context(av.open(str(source)))
            input_stream = reader.streams.video[0]
            writer = resources.enter_context(av.open(str(temporary), mode='w', format='mp4', options={'movflags': '+faststart'}))
            output_stream = writer.add_stream('libx264', rate=original.fps)
            output_stream.width, output_stream.height = original.width, original.height
            output_stream.pix_fmt = 'yuv420p'
            output_stream.time_base = original.time_base
            output_stream.codec_context.time_base = original.time_base
            output_stream.codec_context.max_b_frames = 0
            output_stream.options = {'preset': 'fast', 'crf': '20', 'bf': '0'}
            first = None
            durations = {}

            def mux(packets):
                for packet in packets:
                    timestamp = packet.pts * packet.time_base
                    if timestamp not in durations:
                        raise RuntimeError('Encoder changed the presentation timeline')
                    packet.duration = max(1, round(durations.pop(timestamp) / packet.time_base))
                    writer.mux(packet)

            count = 0
            for frame in reader.decode(input_stream):
                timestamp = frame.pts * frame.time_base
                if first is None:
                    first = timestamp
                relative = timestamp - first
                pixels = frame.to_ndarray(format='bgr24')
                # Every engine sees the same unannotated frame.
                combined = [(spec.label, engine.detect(pixels)) for spec, engine in engines]
                for label, boxes in combined:
                    detections += len(boxes)
                    draw_detections(pixels, label, boxes)
                encoded = av.VideoFrame.from_ndarray(pixels, format='bgr24')
                encoded.pts = int(relative / original.time_base)
                encoded.time_base = original.time_base
                duration = frame.duration * frame.time_base if frame.duration else 1 / original.fps
                durations[relative] = duration
                mux(output_stream.encode(encoded))
                count += 1
                if count % 10 == 0 or count == original.frames:
                    update('Processing frames', count, original.frames)
            mux(output_stream.encode(None))
            if count != original.frames or durations:
                raise RuntimeError('Video processing did not preserve all frames')
        update('Verifying output', 0, original.frames)
        result = inspect_video(temporary, lambda count, total: update('Verifying output', count, total))
        if (result.width, result.height, result.frames) != (original.width, original.height, original.frames):
            raise RuntimeError('Output dimensions or frame count changed')
        if result.timeline_hash != original.timeline_hash:
            raise RuntimeError('Output presentation timestamps changed')
        if abs(result.duration - original.duration) > max(result.time_base, original.time_base):
            raise RuntimeError('Output duration changed')
        if abs(float(result.fps / original.fps) - 1) > 0.001:
            raise RuntimeError('Output playback frame rate changed')
        if result.audio_streams or result.codec != 'h264':
            raise RuntimeError('Output must be silent H.264')
        temporary.replace(destination)
        update('Complete', original.frames, original.frames)
        return original, result, detections
    finally:
        temporary.unlink(missing_ok=True)
