"""Complete timestamp-preserving, silent H.264 video processing."""
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path

import av
import cv2

from .registry import model_specs, DEFAULT_REGISTRY
from .runtime import Detector, Detection


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


def draw_detections(frame, labels, detections, colors=None):
    colors = colors or [(0,255,0)] * len(labels)
    for detection in detections:
        if isinstance(detection, Detection):
            x1,y1,x2,y2,confidence,class_id = detection.x1,detection.y1,detection.x2,detection.y2,detection.confidence,detection.class_id
            if not 0 <= class_id < len(labels) or detection.class_label != labels[class_id]:
                raise ValueError('Detection class ID/label disagrees with model labels')
        else:
            x1,y1,x2,y2,confidence,class_id = detection
        if not float(class_id).is_integer() or not 0 <= int(class_id) < len(labels):
            raise ValueError('Detection class ID disagrees with model labels')
        label = labels[int(class_id)]
        color = colors[int(class_id)]
        left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        text = f'{label} {confidence:.2f}'
        scale = max(0.4, min(frame.shape[1] / 1400, 0.8))
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        text_left = max(0, min(left, frame.shape[1] - text_width - 4))
        text_y = max(text_height + 4, top - 5)
        cv2.rectangle(frame, (text_left, text_y-text_height-3),
                      (text_left+text_width+3, text_y+baseline), (20, 28, 20), -1)
        cv2.putText(frame, text, (text_left+1, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return frame


def pair_footwear(detections, labels):
    """Suppress duplicate shoe boxes and join only unambiguous pairs per frame.

    This is display-only: a pair is not proof that both shoes belong to one
    worker. Uncertain matches remain separate; no identity is retained.
    """
    # shortcut: proximity cannot prove ownership; upgrade with validated pair labels or person association.
    if not any('SHOE' in label.upper() for label in labels):
        return detections
    rows = list(detections)
    def values(item):
        return (item.x1, item.y1, item.x2, item.y2, item.confidence, item.class_id) if isinstance(item, Detection) else item
    shoes = [i for i, box in enumerate(rows) if 'SHOE' in labels[int(values(box)[5])].upper()]

    def overlap(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2-x1) * max(0, y2-y1)
        area_a, area_b = max(0, a[2]-a[0])*max(0, a[3]-a[1]), max(0, b[2]-b[0])*max(0, b[3]-b[1])
        if not intersection or min(area_a, area_b) <= 0:
            return False
        return (intersection / (area_a + area_b - intersection) >= .55
                or intersection / min(area_a, area_b) >= .8)

    # NMS in the display layer also removes duplicate boxes with conflicting
    # classes; preserve the model's highest-confidence classification.
    kept = []
    for i in sorted(shoes, key=lambda index: values(rows[index])[4], reverse=True):
        if not any(overlap(values(rows[i]), values(rows[j])) for j in kept):
            kept.append(i)
    kept.sort()

    neighbors = {}
    for i in kept:
        a = values(rows[i])
        choices = []
        for j in kept:
            if i == j:
                continue
            b = values(rows[j])
            if int(a[5]) != int(b[5]):
                continue
            width = max(a[2]-a[0], b[2]-b[0])
            height = max(a[3]-a[1], b[3]-b[1])
            if min(width, height) <= 0:
                continue
            center_gap = abs((a[0]+a[2]-b[0]-b[2])/2)
            dx = center_gap / width
            dy = abs((a[1]+a[3]-b[1]-b[3])/2) / height
            if .3 <= dx <= 1.6 and dy <= 1.25 and center_gap <= 1.2 * height:
                choices.append((dx + .5*dy, j))
        choices.sort()
        neighbors[i] = choices

    matches = {}
    for i in kept:
        choices = neighbors[i]
        if not choices:
            continue
        score, j = choices[0]
        other = neighbors[j]
        # Require reciprocal nearest neighbors and a clear margin to the next
        # candidate on both sides. Crowded/ambiguous groups stay unpaired.
        if (other and other[0][1] == i
                and (len(choices) == 1 or choices[1][0] > score * 1.25)
                and (len(other) == 1 or other[1][0] > other[0][0] * 1.25)):
            matches[i] = j

    output = []
    consumed = set()
    for i, first in enumerate(rows):
        if i in consumed or (i in shoes and i not in kept):
            continue
        j = matches.get(i)
        if j is not None and j not in consumed:
            a, b = values(first), values(rows[j])
            label = labels[int(a[5])]
            output.append(Detection(min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]),
                                    max(a[3],b[3]), min(a[4],b[4]), int(a[5]), label))
            consumed.add(j)
        else:
            output.append(first)
    return output


def annotate_frame(pixels, engines):
    """Run each detector on original pixels, then draw the combined result."""
    results = [(spec, engine.detect(pixels)) for spec, engine in engines]
    for spec, boxes in results:
        draw_detections(pixels, spec.labels, pair_footwear(boxes, spec.labels),
                        [(0, 220, 0)] * len(spec.labels))
    return results


def process_video(source, destination, models_directory, progress=None, engine_factory=None, statistics=None,
                  registry=DEFAULT_REGISTRY, backend='onnx', device='GPU'):
    """Run every installed model on every original frame, then verify the output.

    engine_factory is injectable for tests; ONNX Runtime is the default backend.
    The only output artifact is the complete annotated video, with no audio.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('Input and output must be different files')
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing output video')
    specifications = model_specs(models_directory, registry, backend, device=device)
    class_counts = {label: 0 for spec in specifications for label in spec.labels}

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
            engines = [(spec, resources.enter_context(
                Detector(spec) if engine_factory is None
                else engine_factory(spec.path))) for spec in specifications]
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
                combined = annotate_frame(pixels, engines)
                for spec, boxes in combined:
                    detections += len(boxes)
                    for box in boxes:
                        class_counts[box.class_label if isinstance(box,Detection) else spec.labels[int(box[5])]] += 1
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
        if statistics is not None:
            statistics.update(class_counts)
        return original, result, detections
    finally:
        temporary.unlink(missing_ok=True)
