from fastapi.testclient import TestClient

from app.config import ConfigStore
from app.inference import Detector
from app.server import create_app


def test_frames_are_finite_uncached_and_identify_the_camera_session(tmp_path, monkeypatch):
    app = create_app(
        store=ConfigStore(tmp_path / "settings.json"),
        detector=Detector(),
        initialize=False,
        scratch_root=tmp_path / "scratch",
    )
    runtime = app.state.runtime
    with TestClient(app) as client:
        assert client.get("/api/frame/upper").status_code == 204
        assert client.get("/api/frame/invalid").status_code == 404
        assert client.get("/api/stream/upper").status_code == 404
        runtime.previews["upper"] = {"sequence": 7}
        runtime.generation = 3
        monkeypatch.setattr(runtime, "frame", lambda _: b"jpeg-frame")
        response = client.get("/api/frame/upper")
        assert response.status_code == 200
        assert response.content == b"jpeg-frame"
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["x-frame-sequence"] == "3:7"
        assert response.headers["cache-control"] == "no-store"
