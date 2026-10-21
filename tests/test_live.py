import time
import unittest
import threading
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from app.live import LiveCameras, available_cameras, open_camera
from app.video import pair_footwear
from app.runtime import Detection


class FakeCapture:
    def __init__(self, value=0):
        self.released = False
        self.value = value
        self.failed = False

    def read(self):
        time.sleep(.005)
        return not self.failed, np.full((48, 64, 3), self.value, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self, spec):
        self.spec = spec
        self.closed = False
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def detect(self, frame):
        self.calls.append((int(frame[0, 0, 0]), threading.get_ident()))
        return [Detection(10, 10, 25, 20, .9, 0, self.spec.labels[0])]


class LiveTests(unittest.TestCase):
    def make_camera(self, opener=None, detector_factory=None):
        self.captures = [FakeCapture(20), FakeCapture(80)]
        self.engines = []
        specs = [SimpleNamespace(labels=(label,), camera_role=role)
                 for role, label in [('face', 'goggles'), ('feet', 'SAFETY_SHOE')]]
        def factory(spec):
            engine = FakeDetector(spec)
            self.engines.append(engine)
            return engine
        camera = LiveCameras(specs, SimpleNamespace(exception=lambda *args: None),
                             opener=opener or (lambda index: self.captures[index]),
                             detector_factory=detector_factory or factory)
        self.addCleanup(camera.stop)
        return camera

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(predicate())

    def test_two_cameras_route_to_correct_models_on_one_inference_thread(self):
        camera = self.make_camera()
        camera.start({'face': 0, 'feet': 1})
        self.wait_for(lambda: all(feed['state'] == 'running' for feed in camera.status()['feeds'].values()))
        for role in ('face', 'feet'):
            part = next(camera.stream(role))
            self.assertIn(b'Content-Type: image/jpeg', part)
            self.assertIsNotNone(cv2.imdecode(np.frombuffer(part.split(b'\r\n\r\n', 1)[1].rstrip(b'\r\n'), np.uint8), 1))
        camera.stop()
        self.assertTrue(all(capture.released for capture in self.captures))
        self.assertTrue(all(engine.closed for engine in self.engines))
        self.assertEqual(len(self.engines), 2)
        self.assertEqual({value for value, _ in self.engines[0].calls}, {20})
        self.assertEqual({value for value, _ in self.engines[1].calls}, {80})
        self.assertEqual(len({thread for engine in self.engines for _, thread in engine.calls}), 1)
        self.assertEqual(camera.status()['state'], 'stopped')

    def test_rejects_invalid_assignment_and_releases_partial_start(self):
        camera = self.make_camera()
        for assignment in (None, [], {}, {'face': None, 'feet': None}, {'face': True, 'feet': 1},
                           {'other': 0}, {'face': -1}, {'feet': '0'}):
            with self.assertRaises(ValueError):
                camera.start(assignment)
        self.captures[1].failed = True
        with self.assertRaisesRegex(ValueError, 'not returning frames'):
            camera.start({'face': 0, 'feet': 1})
        self.assertTrue(all(capture.released for capture in self.captures))
        self.assertFalse(camera.status()['active'])

    def test_each_single_role_only_opens_and_loads_its_own_resources(self):
        for role, other, value in [('face', 'feet', 20), ('feet', 'face', 80)]:
            with self.subTest(role=role):
                camera = self.make_camera()
                camera.start({role: 0 if role == 'face' else 1, other: None})
                self.wait_for(lambda: camera.status()['state'] == 'running')
                camera.stop()
                self.assertEqual(set(camera.status()['feeds']), {role})
                self.assertEqual(len(self.engines), 1)
                self.assertEqual(self.engines[0].spec.camera_role, role)
                self.assertEqual({pixel for pixel, _ in self.engines[0].calls}, {value})
                self.assertFalse(self.captures[1 if role == 'face' else 0].released)

    def test_shared_camera_opens_once_and_both_models_see_original_frame(self):
        opened = []
        def opener(index):
            opened.append(index)
            return self.captures[index]
        camera = self.make_camera(opener=opener)
        camera.start({'face': 0, 'feet': 0})
        self.wait_for(lambda: camera.status()['state'] == 'running')
        camera.stop()
        self.assertEqual(opened, [0])
        self.assertEqual(list(camera.status()['feeds']), ['face'])
        self.assertEqual(camera.status()['feeds']['face']['roles'], ['face', 'feet'])
        self.assertEqual(len(self.engines), 2)
        self.assertTrue(all({pixel for pixel, _ in engine.calls} == {20} for engine in self.engines))
        self.assertTrue(all(engine.closed for engine in self.engines))

    def test_worker_start_failure_releases_opened_devices(self):
        camera = self.make_camera()
        with patch('app.live.threading.Thread.start', side_effect=RuntimeError('thread unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'thread unavailable'):
                camera.start({'face': 0, 'feet': 1})
        self.assertTrue(all(capture.released for capture in self.captures))
        self.assertFalse(camera.status()['active'])

    def test_disconnected_camera_does_not_stop_healthy_camera_or_show_stale_image(self):
        camera = self.make_camera()
        camera.start({'face': 0, 'feet': 1})
        self.wait_for(lambda: all(engine.calls for engine in self.engines) and len(self.engines) == 2)
        self.captures[0].failed = True
        self.wait_for(lambda: camera.status()['feeds']['face']['state'] == 'failed')
        before = len(self.engines[1].calls)
        self.wait_for(lambda: len(self.engines[1].calls) > before)
        self.assertIsNone(camera.feeds['face']['jpeg'])
        self.assertEqual(list(camera.stream('face')), [])
        camera.stop()

    def test_engine_failure_releases_devices_and_allows_restart(self):
        def broken(spec):
            raise RuntimeError('engine failed')
        camera = self.make_camera(detector_factory=broken)
        camera.start({'face': 0, 'feet': 1})
        self.wait_for(lambda: not camera.status()['active'])
        self.assertEqual(camera.status()['state'], 'failed')
        self.assertTrue(all(capture.released for capture in self.captures))
        camera.detector_factory = FakeDetector
        camera.start({'feet': 1, 'face': 0})
        self.wait_for(lambda: camera.status()['state'] == 'running')

    def test_slow_inference_drops_backlog_and_old_stream_ends_after_restart(self):
        camera = self.make_camera()
        def slow(spec):
            engine = FakeDetector(spec)
            original = engine.detect
            def detect(frame):
                time.sleep(.03)
                return original(frame)
            engine.detect = detect
            return engine
        camera.detector_factory = slow
        camera.start({'face': 0, 'feet': 1})
        self.wait_for(lambda: all(feed['state'] == 'running' for feed in camera.status()['feeds'].values()))
        stream = camera.stream('face')
        next(stream)
        camera.stop()
        camera.start({'face': 0, 'feet': 1})
        self.assertEqual(list(stream), [])
        self.assertTrue(all(len(feed['latest']) == 3 for feed in camera.feeds.values()))

    def test_stale_output_is_marked_and_missing_role_rejected(self):
        camera = self.make_camera()
        camera.specs = camera.specs[:1]
        with self.assertRaisesRegex(ValueError, 'feet'):
            camera.start({'face': 0, 'feet': 1})
        with camera.lock:
            camera.feeds['face'] = dict(camera=0, state='running', error=None,
                                        captured=time.monotonic()-4, fps=1, latency_ms=10)
        self.assertEqual(camera.status()['feeds']['face']['state'], 'stale')

    def test_discovery_only_lists_readable_devices(self):
        def opener(index):
            if index == 1:
                return FakeCapture()
            raise ValueError('not present')
        self.assertEqual(available_cameras(opener, 3), [{'index': 1, 'name': 'Camera 1'}])

    def test_windows_discovery_enumerates_without_opening_unused_indices(self):
        device = SimpleNamespace(index=0, name='USB Camera')
        with patch('app.live.sys.platform', 'win32'), patch('app.live._windows_devices', return_value=[device]), \
                patch('app.live.cv2.VideoCapture') as capture:
            self.assertEqual(available_cameras(), [{'index': 0, 'name': 'USB Camera (Camera 0)'}])
            capture.assert_not_called()

    @unittest.skipUnless(__import__('sys').platform == 'win32', 'Windows camera backends')
    def test_windows_backend_fallback_matches_path_not_camera_index(self):
        first = SimpleNamespace(index=0, backend=cv2.CAP_DSHOW, path='usb-camera-a')
        wrong = SimpleNamespace(index=0, backend=cv2.CAP_MSMF, path='usb-camera-b')
        match = SimpleNamespace(index=1, backend=cv2.CAP_MSMF, path='USB-CAMERA-A')
        class Capture:
            def __init__(self, opened): self.opened, self.released = opened, False
            def isOpened(self): return self.opened
            def set(self, *args): pass
            def release(self): self.released = True
        failed, usable = Capture(False), Capture(True)
        with patch('app.live._windows_devices', return_value=[first]), \
                patch('cv2_enumerate_cameras.enumerate_cameras', return_value=[wrong, match]), \
                patch('app.live.cv2.VideoCapture', side_effect=[failed, usable]) as capture:
            self.assertIs(open_camera(0), usable)
            self.assertEqual([call.args for call in capture.call_args_list], [(0, cv2.CAP_DSHOW), (1, cv2.CAP_MSMF)])
            self.assertTrue(failed.released)

    def test_shoe_pair_display_keeps_other_classes_and_unpaired_shoes(self):
        labels = ('SAFETY_SHOE', 'NOT_SAFETY_SHOE')
        boxes = [Detection(10, 40, 20, 55, .9, 0, labels[0]),
                 Detection(24, 41, 34, 56, .8, 0, labels[0]),
                 Detection(50, 40, 60, 55, .7, 1, labels[1])]
        result = pair_footwear(boxes, labels)
        self.assertEqual(len(result), 2)
        self.assertEqual((result[0].x1, result[0].x2, result[0].confidence), (10, 34, .8))
        self.assertEqual(result[1], boxes[2])

    def test_shoe_pair_display_drops_duplicates_and_does_not_cross_workers(self):
        label = 'SAFETY_SHOE'
        boxes = [Detection(124, 657, 168, 707, .54, 0, label),
                 Detection(122, 653, 186, 708, .34, 0, label),
                 Detection(182, 762, 266, 842, .44, 0, label),
                 Detection(227, 697, 296, 760, .42, 0, label),
                 Detection(363, 673, 417, 730, .53, 0, label),
                 Detection(366, 681, 406, 729, .35, 0, label)]
        result = pair_footwear(boxes, (label,))
        self.assertEqual(len(result), 3)
        self.assertEqual([(round(box.x1), round(box.x2)) for box in result],
                         [(124, 168), (182, 296), (363, 417)])

    def test_shoe_pair_display_leaves_close_workers_separate(self):
        label = 'SAFETY_SHOE'
        boxes = [Detection(133, 592, 173, 627, .38, 0, label),
                 Detection(195, 618, 241, 665, .34, 0, label),
                 Detection(339, 554, 364, 591, .33, 0, label),
                 Detection(367, 604, 408, 650, .50, 0, label)]
        result = pair_footwear(boxes, (label,))
        self.assertEqual(len(result), 3)
        self.assertEqual((result[0].x1, result[0].x2), (133, 173))
        self.assertEqual((result[-1].x1, result[-1].x2), (339, 408))

    def test_shoe_pair_display_preserves_class_and_ambiguous_boxes(self):
        labels = ('SAFETY_SHOE', 'NOT_SAFETY_SHOE')
        boxes = [Detection(10, 40, 20, 55, .9, 0, labels[0]),
                 Detection(11, 40, 21, 55, .4, 1, labels[1]),
                 Detection(23, 40, 33, 55, .8, 0, labels[0]),
                 Detection(36, 40, 46, 55, .7, 0, labels[0])]
        result = pair_footwear(boxes, labels)
        self.assertEqual(len(result), 3)
        self.assertEqual([box.confidence for box in result], [.9, .8, .7])
        self.assertTrue(all(box.class_label == labels[0] for box in result))


if __name__ == '__main__':
    unittest.main()
