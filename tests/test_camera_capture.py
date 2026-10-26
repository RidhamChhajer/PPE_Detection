import cv2
import numpy as np
import pytest

from app.cameras import Camera, CameraPool
from app.config import CameraProfile, Settings, ViewConfig
from tests.test_core import FakeCapture, wait_until


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_capture_recovers_from_empty_warmup_and_normalizes_frames(channels):
    class WarmingCamera(FakeCapture):
        reads = 0

        def read(self):
            self.reads += 1
            if self.reads <= 2:
                return True, None
            super().read()
            return True, np.full((240, 320, channels), 100, np.uint8)

    capture = WarmingCamera()
    camera = Camera(0, CameraProfile(), lambda _: capture)
    camera.start()
    try:
        wait_until(lambda: camera.snapshot()[3]["connected"])
        frame, _, _, info = camera.snapshot()
        assert frame.shape == (240, 320, 3)
        assert frame[0, 0].tolist() == [100, 100, 100]
        assert info["message"] == ""
        assert cv2.CAP_PROP_FOURCC not in capture.values
    finally:
        assert camera.stop()
    assert capture.released


def test_camera_reopens_after_disconnect():
    class Disconnected(FakeCapture):
        def read(self):
            return False, None

    first, recovered = Disconnected(), FakeCapture()
    opened = []

    def factory(_):
        capture = first if not opened else recovered
        opened.append(capture)
        return capture

    camera = Camera(0, CameraProfile(), factory)
    camera.start()
    try:
        wait_until(lambda: camera.snapshot()[3]["connected"], seconds=8)
        assert first.released
        assert opened == [first, recovered]
    finally:
        assert camera.stop()
    assert recovered.released


def test_missing_named_camera_is_rejected_before_starting_any_capture(monkeypatch):
    monkeypatch.setattr("app.cameras.camera_catalog", lambda: [{"id": "present", "index": 0}])
    pool = CameraPool()
    with pytest.raises(ValueError, match="Selected upper camera is not connected"):
        pool.start(Settings(upper=ViewConfig(camera="unplugged")))
    assert pool.devices == {}


def test_start_retries_transiently_missing_driver_limits_without_discarding_controls(monkeypatch):
    monkeypatch.setattr("app.cameras.camera_catalog", lambda: [{"id": "present", "index": 0}])
    monkeypatch.setattr("app.cameras.open_camera", lambda _: FakeCapture())
    reports = []

    def capabilities(_):
        reports.append(1)
        controls = (
            {}
            if len(reports) < 3
            else {"auto_exposure": {"min": 0, "max": 1, "step": 1, "current": 1}}
        )
        return {"available": True, "controls": controls}

    settings = Settings(upper=ViewConfig(camera="present"), lower=ViewConfig(camera=None))
    settings.profiles["present"] = CameraProfile(controls={"auto_exposure": 1})
    pool = CameraPool()
    pool.capability_provider = capabilities
    try:
        pool.start(settings)
        wait_until(lambda: pool.devices["present"].snapshot()[3]["connected"])
        assert len(reports) >= 3
        assert settings.profiles["present"].controls == {"auto_exposure": 1}
    finally:
        assert pool.stop()
