import threading
import time

from app.cameras import CameraPool
from app.config import ConfigStore
from app.runtime import Runtime
from tests.test_core import FakeCapture, FakeDetector, wait_until


def test_video_advances_while_inference_is_blocked_and_does_not_queue(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class SlowDetector(FakeDetector):
        def predict(self, *args):
            entered.set()
            release.wait(timeout=5)
            return super().predict(*args)

    runtime = Runtime(
        ConfigStore(tmp_path / "settings.json"),
        SlowDetector(),
        CameraPool(lambda _: FakeCapture()),
    )
    runtime.start()
    try:
        assert entered.wait(timeout=2)
        wait_until(lambda: "upper" in runtime.previews)
        first = runtime.previews["upper"]["sequence"]
        jpeg = runtime.frame("upper")
        assert jpeg.startswith(b"\xff\xd8")
        wait_until(lambda: runtime.previews["upper"]["sequence"] > first + 5)
        with runtime.lock:
            latest = runtime.previews["upper"]["sequence"]
            image = runtime.frame("upper")
            assert latest > first + 5
            assert image is runtime.previews["upper"]["jpeg"]
        assert image.startswith(b"\xff\xd8")
        state = runtime.snapshot()["views"]["upper"]
        assert state["connected"] and state["stale"]
        assert state["preview_age_ms"] < 500
        assert state["preview_fps"] > 0
        assert state["detections"] == []
        assert state["summary"]["goggles"] == "unknown"
        old_generation = runtime.generation
    finally:
        release.set()
        runtime.stop()
    assert runtime.frame("upper") is runtime.empty
    runtime.start()
    try:
        assert runtime.generation != old_generation
    finally:
        runtime.stop()


def test_http_frame_reads_do_not_encode_or_run_inference(tmp_path, monkeypatch):
    runtime = Runtime(
        ConfigStore(tmp_path / "settings.json"),
        FakeDetector(),
        CameraPool(lambda _: FakeCapture()),
    )
    runtime.start()
    try:
        wait_until(lambda: "upper" in runtime.previews)
        before = runtime.frame("upper")

        def forbidden(*_):
            raise AssertionError("HTTP frame reads must not encode video")

        with runtime.lock:
            monkeypatch.setattr("app.runtime.jpeg", forbidden)
            assert runtime.frame("upper") is before
            runtime.states["upper"] = {"timestamp": time.monotonic() - 5, "detections": []}
            assert runtime.frame("upper") is before
            runtime.running = False
            assert runtime.frame("upper") is runtime.empty
    finally:
        runtime.stop()
