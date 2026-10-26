import cv2
import numpy as np
import av
from fastapi.testclient import TestClient

from app.config import ConfigStore
from app.media import MediaTest
from app.server import create_app
from tests.test_core import FakeDetector, wait_until


class RecordingDetector(FakeDetector):
    def __init__(self):
        super().__init__()
        self.calls = []

    def predict(self, view, frame, cfg):
        self.calls.append(view)
        return super().predict(view, frame, cfg)


def video_bytes(path, frames=3):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((120, 160, 3), np.uint8))
    writer.release()
    return path.read_bytes()


def test_combined_image_runs_both_models_and_never_creates_video(tmp_path):
    detector = RecordingDetector()
    media = MediaTest(detector, ConfigStore(tmp_path / "settings.json"), tmp_path / "exports")
    source = tmp_path / "image.png"
    cv2.imwrite(str(source), np.zeros((120, 160, 3), np.uint8))
    media.start(source, "both")
    wait_until(lambda: media.snapshot()["state"] != "running")
    media.thread.join()
    state = media.snapshot()
    assert state["state"] == "complete"
    assert detector.calls == ["upper", "lower"]
    assert state["summary"]["goggles"] == "detected"
    assert state["summary"]["shoes"] == "detected"
    assert not state["download_ready"]
    assert not source.exists()
    media.stop(clear=True)


def test_combined_annotated_video_download_and_cleanup(tmp_path):
    detector = RecordingDetector()
    app = create_app(
        store=ConfigStore(tmp_path / "settings.json"),
        detector=detector,
        initialize=False,
        scratch_root=tmp_path / "scratch",
    )
    with TestClient(app) as client:
        payload = video_bytes(tmp_path / "input.avi")
        response = client.post(
            "/api/test/both",
            files={"file": ("input.avi", payload, "video/x-msvideo")},
        )
        assert response.status_code == 200
        job = response.json()["id"]
        wait_until(lambda: app.state.media.snapshot()["state"] != "running")
        app.state.media.thread.join()
        state = client.get("/api/test").json()
        assert state["state"] == "complete", state
        assert state["download_ready"]
        assert detector.calls == ["upper", "lower"] * 3
        download = client.get(f"/api/test/{job}/download")
        assert download.status_code == 200
        assert download.headers["content-type"] == "video/mp4"
        assert "attachment" in download.headers["content-disposition"]
        inline = client.get(f"/api/test/{job}/video", headers={"Range": "bytes=0-99"})
        assert inline.status_code == 206
        assert inline.content == download.content[:100]
        assert inline.headers["content-disposition"].startswith("inline;")
        assert (
            client.get(f"/api/test/{job}/video", headers={"Range": "bytes=999999999-"}).status_code
            == 416
        )
        output = tmp_path / "download.mp4"
        output.write_bytes(download.content)
        with av.open(str(output)) as encoded:
            assert encoded.streams.video[0].codec_context.name == "h264"
            assert not encoded.streams.audio
        cap = cv2.VideoCapture(str(output))
        ok, frame = cap.read()
        assert ok and frame.shape[:2] == (120, 160)
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 3
        assert frame[:30, :30].max() > 30  # Colored annotations are actually encoded.
        cap.release()
        assert client.get("/api/test/wrong-id/download").status_code == 404
        assert client.delete("/api/test").status_code == 200
        assert client.get(f"/api/test/{job}/download").status_code == 404
        assert not list(app.state.media.output_dir.glob("ppe-*.mp4"))


def test_combined_requires_both_models_and_video_export_is_automatic(tmp_path):
    detector = RecordingDetector()
    detector.models.pop("lower")
    app = create_app(
        store=ConfigStore(tmp_path / "settings.json"),
        detector=detector,
        initialize=False,
        scratch_root=tmp_path / "scratch",
    )
    with TestClient(app) as client:
        assert client.post("/api/test/both", files={"file": ("a.jpg", b"bad")}).status_code == 409
        assert not detector.calls
    detector.models["lower"] = True
    media = MediaTest(detector, ConfigStore(tmp_path / "other.json"), tmp_path / "exports")
    source = tmp_path / "clip.avi"
    video_bytes(source)
    media.start(source, "both")
    wait_until(lambda: media.snapshot()["state"] != "running")
    media.thread.join()
    assert media.snapshot()["state"] == "complete"
    assert media.snapshot()["download_ready"] and media.output is not None
