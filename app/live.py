"""One capture per physical camera and one fair, context-owning inference worker."""
from contextlib import ExitStack
import sys
import threading
import time

import cv2

from .runtime import Detector
from .video import annotate_frame

ROLES = ('face', 'feet')
MAX_FRAME_AGE = 3.0


def open_camera(index):
    devices = _windows_devices() if sys.platform == 'win32' else []
    if sys.platform == 'win32':
        selected = next((device for device in devices if device.index == index), None)
        if selected is None:
            raise ValueError('Selected camera is no longer connected. Refresh the camera list.')
        choices = [(selected.index, selected.backend)]
        # Match device paths, not indices: Windows backends can order cameras differently.
        if selected.path:
            from cv2_enumerate_cameras import enumerate_cameras
            other = cv2.CAP_MSMF if selected.backend == cv2.CAP_DSHOW else cv2.CAP_DSHOW
            choices += [(device.index, device.backend) for device in enumerate_cameras(other)
                        if device.path and device.path.casefold() == selected.path.casefold()]
    else:
        choices = [(index, cv2.CAP_ANY)]
    for device_index, backend in choices:
        capture = cv2.VideoCapture(device_index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        capture.release()
    raise ValueError(f'Camera {index} could not be opened. Close other camera apps and check Windows camera permissions.')


def _windows_devices():
    from cv2_enumerate_cameras import enumerate_cameras
    devices = list(enumerate_cameras(cv2.CAP_DSHOW))
    return devices or list(enumerate_cameras(cv2.CAP_MSMF))


def available_cameras(opener=open_camera, limit=8):
    if sys.platform == 'win32' and opener is open_camera:
        return [dict(index=device.index, name=f'{device.name} (Camera {device.index})')
                for device in _windows_devices()]
    cameras = []
    for index in range(limit):
        try:
            capture = opener(index)
            try:
                ok, _ = capture.read()
                if ok:
                    cameras.append({'index': index, 'name': f'Camera {index}'})
            finally:
                capture.release()
        except (ValueError, cv2.error):
            continue
    return cameras


class LiveCameras:
    def __init__(self, specs, logger, opener=open_camera, detector_factory=Detector):
        self.specs, self.logger = specs, logger
        self.opener, self.detector_factory = opener, detector_factory
        self.lock = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = None
        self.generation = 0
        self.state = 'stopped'
        self.error = None
        self.feeds = {}

    def start(self, cameras):
        if (not isinstance(cameras, dict) or not cameras or set(cameras) - set(ROLES)
                or any(index is not None and (type(index) is not int or not 0 <= index < 100)
                       for index in cameras.values())):
            raise ValueError('Select at least one camera from the list')
        cameras = {role: index for role, index in cameras.items() if index is not None}
        if not cameras:
            raise ValueError('Select at least one camera from the list')
        for role in cameras:
            if not any(spec.camera_role == role for spec in self.specs):
                raise ValueError(f'No detector is installed for the {role} camera')
        with self.lock:
            if self.thread is not None:
                raise RuntimeError('Stop the active cameras before starting again')
            captures, initial, groups = {}, {}, {}
            for role, index in cameras.items():
                primary = next((key for key in groups if cameras[key] == index), role)
                groups.setdefault(primary, []).append(role)
            try:
                for role in groups:
                    index = cameras[role]
                    capture = captures[role] = self.opener(index)
                    ok, pixels = capture.read()
                    if not ok:
                        raise ValueError(f'{role.title()} camera is not returning frames')
                    initial[role] = (1, time.monotonic(), pixels)
            except Exception:
                for capture in captures.values():
                    capture.release()
                raise
            self.generation += 1
            self.feeds = {role: dict(camera=cameras[role], roles=groups[role], state='starting', error=None,
                                    latest=initial[role], jpeg=None, sequence=0,
                                    captured=None, fps=0.0, latency_ms=0.0)
                          for role in groups}
            self.error = None
            self.state = 'starting'
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, args=(captures,), daemon=True, name='ppe-inference')
            try:
                self.thread.start()
            except Exception:
                self.thread = None
                self.state = 'failed'
                self.stop_event.set()
                for capture in captures.values():
                    capture.release()
                raise

    def _capture(self, role, capture):
        sequence = 1
        try:
            while not self.stop_event.is_set():
                ok, pixels = capture.read()
                if not ok:
                    raise RuntimeError('Camera disconnected. Stop cameras, reconnect, and start again.')
                sequence += 1
                with self.lock:
                    self.feeds[role]['latest'] = (sequence, time.monotonic(), pixels)
                    self.lock.notify_all()
        except Exception as error:
            self.logger.exception('%s camera stopped', role)
            with self.lock:
                self.feeds[role].update(state='failed', error=str(error), latest=None, jpeg=None)
                self.lock.notify_all()
        finally:
            capture.release()

    def _run(self, captures):
        workers = {}
        try:
            with ExitStack() as stack:
                engines = {role: [(spec, stack.enter_context(self.detector_factory(spec)))
                                  for spec in self.specs if spec.camera_role in self.feeds[role]['roles']]
                           for role in self.feeds}
                for role, capture in captures.items():
                    worker = threading.Thread(target=self._capture, args=(role, capture), daemon=True,
                                              name=f'ppe-capture-{role}')
                    worker.start()
                    workers[role] = worker
                seen = dict.fromkeys(self.feeds, 0)
                last = dict.fromkeys(self.feeds, time.monotonic())
                while not self.stop_event.is_set():
                    worked = False
                    for role in self.feeds:
                        if self.stop_event.is_set():
                            break
                        with self.lock:
                            latest = self.feeds[role]['latest']
                        if latest is None or latest[0] == seen[role]:
                            continue
                        sequence, captured, pixels = latest
                        seen[role] = sequence
                        if time.monotonic() - captured > MAX_FRAME_AGE:
                            continue
                        worked = True
                        pixels = pixels.copy()
                        annotate_frame(pixels, engines[role])
                        ok, jpeg = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 82])
                        if not ok:
                            raise RuntimeError(f'Could not encode the {role} camera preview')
                        now = time.monotonic()
                        with self.lock:
                            feed = self.feeds[role]
                            if feed['state'] != 'failed' and not self.stop_event.is_set():
                                feed.update(jpeg=jpeg.tobytes(), sequence=sequence, captured=captured,
                                            fps=1 / max(now-last[role], 1e-6),
                                            latency_ms=(now-captured)*1000, state='running')
                                self.state = 'running'
                                self.lock.notify_all()
                        last[role] = now
                    if not worked:
                        with self.lock:
                            self.lock.wait(timeout=.05)
        except Exception as error:
            self.logger.exception('Live inference stopped')
            with self.lock:
                self.error = str(error)
                self.state = 'failed'
                for feed in self.feeds.values():
                    feed.update(state='failed', error=str(error), jpeg=None)
        finally:
            self.stop_event.set()
            # Keep the session reserved until any blocking camera reads return.
            for worker in workers.values():
                worker.join()
            for role, capture in captures.items():
                if role not in workers:
                    capture.release()
            with self.lock:
                if self.state != 'failed':
                    self.state = 'stopped'
                for feed in self.feeds.values():
                    feed['jpeg'] = None
                    if feed['state'] != 'failed':
                        feed['state'] = 'stopped'
                self.thread = None
                self.lock.notify_all()

    def stop(self):
        with self.lock:
            thread = self.thread
            if thread is not None:
                self.state = 'stopping'
            self.stop_event.set()
            self.lock.notify_all()
        if thread is not None:
            thread.join(timeout=5)
        with self.lock:
            if self.thread is not None:
                raise RuntimeError('Camera is still stopping; wait and retry')

    def status(self):
        with self.lock:
            feeds = {}
            for role, feed in self.feeds.items():
                age = None if feed['captured'] is None else (time.monotonic()-feed['captured'])*1000
                state = feed['state']
                if state == 'running' and age > MAX_FRAME_AGE*1000:
                    state = 'stale'
                feeds[role] = dict(camera=feed['camera'], state=state, error=feed['error'],
                                   roles=feed.get('roles', [role]),
                                   fps=round(feed['fps'], 1), latency_ms=round(feed['latency_ms'], 1),
                                   frame_age_ms=None if age is None else round(age, 1))
            return dict(state=self.state, active=self.thread is not None, generation=self.generation,
                        error=self.error, feeds=feeds)

    def stream(self, role):
        with self.lock:
            generation = self.generation
        sequence = 0
        while True:
            with self.lock:
                self.lock.wait_for(lambda: self.generation != generation or self.stop_event.is_set()
                                   or self.feeds[role]['sequence'] != sequence
                                   or self.feeds[role]['state'] == 'failed', timeout=1)
                if (self.generation != generation or self.stop_event.is_set()
                        or self.feeds[role]['state'] == 'failed'):
                    return
                feed = self.feeds[role]
                if feed['sequence'] == sequence:
                    continue
                sequence = feed['sequence']
                jpeg = feed['jpeg']
                if jpeg is None or time.monotonic()-feed['captured'] > MAX_FRAME_AGE:
                    continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
