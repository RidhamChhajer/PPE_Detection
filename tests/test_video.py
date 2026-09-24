from fractions import Fraction
import hashlib
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np

from app.video import inspect_video, process_video


def make_video(path, timestamps):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=25)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = 'yuv420p'
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.max_b_frames = 0
        for index, timestamp in enumerate(timestamps):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 80+index, dtype=np.uint8), format='bgr24')
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


class FakeEngine:
    calls = []

    def __init__(self, path):
        self.label = path.stem

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def detect(self, frame):
        # No previous detector's green annotation may enter another model.
        if np.any((frame[:, :, 1] == 255) & (frame[:, :, 0] == 0)):
            raise AssertionError('Engine received an already-annotated image')
        self.calls.append(self.label)
        return np.array([[10, 20, 20, 30, .9]], dtype=np.float32)


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / 'models'
        self.models.mkdir()
        for label in ('first', 'second'):
            (self.models / f'{label}.engine').write_bytes(b'test')
            digest = hashlib.sha256(b'test').hexdigest()
            (self.models / f'{label}.toml').write_text(f'label="{label}"\nstatus="sample"\nengine_sha256="{digest}"')
        FakeEngine.calls = []

    def test_complete_constant_and_variable_timing(self):
        for name, timestamps in [('cfr', [0, 40, 80, 120]), ('vfr', [0, 40, 140, 180])]:
            with self.subTest(name=name):
                source = self.root / f'{name}.mp4'
                make_video(source, timestamps)
                before = len(FakeEngine.calls)
                original, result, detections = process_video(source, self.root/f'{name}_out.mp4', self.models, engine_factory=FakeEngine)
                self.assertEqual(result.frames, 4)
                self.assertEqual(original.timeline_hash, result.timeline_hash)
                self.assertEqual(original.duration, result.duration)
                self.assertEqual((result.width, result.height), (64,48))
                self.assertEqual(result.audio_streams, 0)
                self.assertEqual(result.codec, 'h264')
                self.assertEqual(detections, 8)
                self.assertEqual(len(FakeEngine.calls)-before, 8)

    def test_failure_removes_partial_output(self):
        class BrokenEngine(FakeEngine):
            def detect(self, frame):
                raise RuntimeError('Injected inference failure')
        source = self.root/'input.mp4'
        make_video(source, [0,40])
        destination = self.root/'output.mp4'
        with self.assertRaisesRegex(RuntimeError, 'Injected'):
            process_video(source, destination, self.models, engine_factory=BrokenEngine)
        self.assertFalse(destination.exists())
        self.assertFalse((self.root/'output.partial.mp4').exists())
        self.assertTrue(source.exists())

    def test_refuses_to_overwrite_input(self):
        source = self.root/'input.mp4'
        with self.assertRaisesRegex(ValueError, 'different'):
            process_video(source, source, self.models, engine_factory=FakeEngine)


if __name__ == '__main__':
    unittest.main()
