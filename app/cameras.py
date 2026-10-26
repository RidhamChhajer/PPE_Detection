import threading
from collections import deque
import time
import math

import cv2

from app.config import CameraProfile
from app.camera_capabilities import validate_controls
from app.camera_driver import DriverCapture

CONTROL_IDS = {
    name: getattr(cv2, "CAP_PROP_" + name.upper())
    for name in (
        "brightness",
        "contrast",
        "saturation",
        "exposure",
        "auto_exposure",
        "focus",
        "autofocus",
        "gain",
    )
}


def frame_rate(samples, now):
    samples.append(now)
    while len(samples) > 1 and now - samples[0] > 1:
        samples.popleft()
    elapsed = now - samples[0]
    return round((len(samples) - 1) / elapsed, 1) if elapsed > 0 else None


def camera_catalog():
    try:
        from cv2_enumerate_cameras import enumerate_cameras

        return [
            {"id": c.path, "index": c.index, "name": c.name}
            for c in enumerate_cameras(cv2.CAP_DSHOW)
            if c.path
        ]
    except (ImportError, OSError, RuntimeError):
        return []


def open_camera(selection):
    return DriverCapture(selection)


class Camera:
    """One owner per USB device; readers never queue frames or touch the capture handle."""

    def __init__(self, index, profile, factory=None, capabilities=None, capability_reader=None):
        self.index = index
        self.profile = profile
        self.factory = factory or open_camera
        self.capabilities, self.capability_reader = capabilities, capability_reader
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"camera-{index}", daemon=True)
        self.frame = None
        self.timestamp = 0.0
        self.sequence = 0
        self.message = "Connecting camera"
        self.defaults = {}
        self.controls = {}
        self.actual = {}
        self.capture = None

    def start(self):
        self.thread.start()

    def _configure(self, capture):
        if self.capabilities is not None:
            validate_controls(self.profile.model_dump(), self.capabilities)
        # Let the driver negotiate its native format; virtual cameras may reject forced MJPG.
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.profile.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.profile.height)
        capture.set(cv2.CAP_PROP_FPS, self.profile.fps)

        def readable(name, value):
            if not math.isfinite(value) or (value == -1 and name != "exposure"):
                return None
            return value if name not in {"auto_exposure", "autofocus"} or value in (0, 1) else None

        defaults = {k: readable(k, capture.get(v)) for k, v in CONTROL_IDS.items()}
        controls = {}
        for name in (
            "auto_exposure",
            "autofocus",
            "brightness",
            "contrast",
            "saturation",
            "exposure",
            "focus",
            "gain",
        ):
            prop = CONTROL_IDS[name]
            target = self.profile.controls.get(name)
            automatic = {"exposure": "auto_exposure", "focus": "autofocus"}.get(name)
            if automatic and self.profile.controls.get(automatic) == 1:
                target = None
            accepted = capture.set(prop, target) if target is not None else None
            actual = readable(name, capture.get(prop))
            controls[name] = {
                "value": actual,
                "requested": target,
                "accepted": accepted,
                "matched": target is None or (actual is not None and abs(actual - target) < 0.05),
            }
        # Do not open a second DirectShow graph from the service while capture owns the device.
        # Isolated captures use OpenCV read-back; unsupported values stay unknown.
        if self.capability_reader is not None and not isinstance(capture, DriverCapture):
            reported = self.capability_reader(self.index)
            if reported.get("available"):
                self.capabilities = reported
                for name, control in controls.items():
                    current = reported["controls"].get(name, {}).get("current")
                    if current is not None:
                        control["value"] = current
                        if control["requested"] is not None:
                            control["accepted"] = control["matched"] = (
                                current == control["requested"]
                            )
        with self.lock:
            if not self.defaults:
                self.defaults = {k: v for k, v in defaults.items() if v is not None}
            self.controls = controls
            self.actual = {
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "requested_fps": self.profile.fps,
                "driver_fps": capture.get(cv2.CAP_PROP_FPS),
            }

    def _run(self):
        while not self.stop_event.is_set():
            capture = None
            try:
                capture = self.factory(self.index)
                with self.lock:
                    self.capture = capture
                if self.stop_event.is_set():
                    break
                if not capture.isOpened():
                    raise RuntimeError(
                        "Camera unavailable. Check USB connection and camera assignment."
                    )
                self._configure(capture)
                samples = deque()
                last_usable = time.monotonic()
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None or not frame.size:
                        if time.monotonic() - last_usable > 5:
                            raise RuntimeError(
                                "Camera returned no usable frames. Check USB access and close other camera apps. Retrying."
                            )
                        self.stop_event.wait(0.1)
                        continue
                    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.ndim == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    elif frame.ndim != 3 or frame.shape[2] != 3:
                        raise RuntimeError(
                            "Unsupported camera image format. Select another camera mode."
                        )
                    last_usable = time.monotonic()
                    now = time.monotonic()
                    with self.lock:
                        self.frame, self.timestamp = frame, now
                        self.sequence += 1
                        self.message = ""
                        self.actual.update(width=frame.shape[1], height=frame.shape[0])
                        self.actual["measured_fps"] = frame_rate(samples, now)
            except Exception as exc:
                with self.lock:
                    self.frame = None
                    self.message = str(exc)
            finally:
                if capture is not None:
                    capture.release()
                with self.lock:
                    self.capture = None
            self.stop_event.wait(1)
        with self.lock:
            self.frame = None
            self.message = "Stopped"

    def snapshot(self):
        with self.lock:
            fresh = self.frame is not None and time.monotonic() - self.timestamp <= 1
            return (
                self.frame if fresh else None,
                self.timestamp,
                self.sequence,
                {
                    "camera": self.index,
                    "connected": fresh,
                    "message": self.message if fresh else self.message or "Camera frames are stale",
                    "actual": dict(self.actual),
                    "controls": dict(self.controls),
                    "defaults": dict(self.defaults),
                    "capabilities": self.capabilities,
                },
            )

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            with self.lock:
                capture = self.capture
            if isinstance(capture, DriverCapture):
                capture.interrupt()
            self.thread.join(timeout=2)
        return not self.thread.is_alive()


class CameraPool:
    def __init__(self, factory=None):
        self.devices = {}
        self.factory = factory
        self.capability_provider = None

    def start(self, settings):
        if not self.stop():
            raise RuntimeError(
                "A camera driver is still releasing a device. Retry in a few seconds."
            )
        catalog = camera_catalog() if self.factory is None else []
        if self.factory is None:
            for view in ("upper", "lower"):
                selection = getattr(settings, view).camera
                if isinstance(selection, str) and not any(c["id"] == selection for c in catalog):
                    raise ValueError(
                        f"Selected {view} camera is not connected. Reconnect it or choose a connected camera in Settings."
                    )
        shared, planned = {}, {}
        for index in {settings.upper.camera, settings.lower.camera} - {None}:
            device_id = next((c["id"] for c in catalog if c["index"] == index), index)
            if device_id in shared:
                planned[index] = shared[device_id]
                continue
            profile = settings.profiles.get(
                str(device_id), settings.profiles.get(str(index), CameraProfile())
            )
            capabilities = self.capability_provider(device_id) if self.capability_provider else None
            if capabilities is not None:
                # This driver briefly omits controls while powering back up after release.
                deadline = time.monotonic() + 3
                while (
                    profile.controls.keys() - capabilities.get("controls", {}).keys()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.25)
                    capabilities = self.capability_provider(device_id)
                validate_controls(profile.model_dump(), capabilities)
            camera = Camera(
                device_id, profile, self.factory, capabilities, self.capability_provider
            )
            planned[index] = camera
            shared[device_id] = camera
        self.devices = planned
        for camera in set(planned.values()):
            camera.start()

    def stop(self):
        stopped = {camera: camera.stop() for camera in set(self.devices.values())}
        remaining = {index: camera for index, camera in self.devices.items() if not stopped[camera]}
        self.devices = remaining
        return not remaining
