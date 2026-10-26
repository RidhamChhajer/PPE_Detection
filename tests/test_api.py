import re
import time

from fastapi.testclient import TestClient

from app.config import ConfigStore, Settings
from app.inference import Detector
from app.server import create_app
from tests.test_core import FakeDetector


def make_client(tmp_path, detector=None):
    return TestClient(
        create_app(
            store=ConfigStore(tmp_path / "settings.json"),
            detector=detector or Detector(),
            initialize=False,
            scratch_root=tmp_path / "scratch",
        )
    )


def test_settings_read_save_validate_and_cross_origin(tmp_path):
    with make_client(tmp_path) as client:
        settings = client.get("/api/settings").json()
        settings["lower"]["camera"] = 0
        assert client.put("/api/settings", json=settings).status_code == 200
        assert client.get("/api/settings").json()["lower"]["camera"] == 0
        settings["upper"]["confidence"] = -1
        assert client.put("/api/settings", json=settings).status_code == 422
        assert client.get("/api/settings").json()["upper"]["confidence"] == 0.35
        assert (
            client.put(
                "/api/settings",
                json=Settings().model_dump(),
                headers={"Origin": "https://evil.example"},
            ).status_code
            == 403
        )
        assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 403


def test_camera_controls_use_driver_ranges_and_device_identity(tmp_path):
    caps = dict(
        available=True,
        device_id="a" * 64,
        message="Test driver",
        controls={"brightness": dict(min=0, max=255, step=5, default=125, current=140)},
    )
    app = create_app(
        store=ConfigStore(tmp_path / "settings.json"),
        detector=FakeDetector(),
        initialize=False,
        scratch_root=tmp_path / "scratch",
        capability_provider=lambda _: caps,
    )
    with TestClient(app) as client:
        assert (
            client.get("/api/camera-capabilities?camera=0").json()["controls"]["brightness"]["max"]
            == 255
        )
        settings = Settings().model_dump()
        settings["profiles"]["0"] = dict(
            width=1280, height=720, fps=30, controls={"brightness": 545}
        )
        assert client.put("/api/settings", json=settings).status_code == 422
        settings["profiles"]["0"]["controls"] = {"brightness": 141}
        assert client.put("/api/settings", json=settings).status_code == 422
        settings["profiles"]["0"]["controls"] = {"gain": 10}
        assert client.put("/api/settings", json=settings).status_code == 422
        settings["profiles"]["0"]["controls"] = {"brightness": 140}
        response = client.put("/api/settings", json=settings)
        assert response.status_code == 200
        saved = response.json()["settings"]
        assert saved["profiles"]["0"]["device_id"] == "a" * 64
        caps["device_id"] = "b" * 64
        assert client.put("/api/settings", json=saved).status_code == 422


def test_missing_model_unknown_view_and_unsupported_file(tmp_path):
    with make_client(tmp_path) as client:
        state = client.get("/api/status").json()
        assert not state["gpu"]["ready"]
        assert state["views"]["upper"]["summary"]["goggles"] == "unknown"
        assert client.get("/api/frame/upper").status_code == 204
        assert client.get("/api/frame/wrong").status_code == 404
        assert (
            client.post("/api/test/upper", files={"file": ("bad.exe", b"abc")}).status_code == 415
        )
        assert (
            client.post("/api/test/upper", files={"file": ("test.jpg", b"abc")}).status_code == 409
        )


def test_invalid_image_cleanup_and_remove(tmp_path):
    detector = FakeDetector()
    with make_client(tmp_path, detector) as client:
        before = set((tmp_path / "scratch/uploads").glob("ppe-*"))
        assert (
            client.post(
                "/api/test/upper", files={"file": ("fake.jpg", b"not an image")}
            ).status_code
            == 200
        )
        for _ in range(100):
            state = client.get("/api/test").json()
            if state["state"] == "error":
                break
            time.sleep(0.01)
        assert state["state"] == "error"
        assert "decoded" in state["message"]
        assert set((tmp_path / "scratch/uploads").glob("ppe-*")) == before
        assert client.delete("/api/test").json()["state"] == "idle"


def test_no_camera_assignment_start_is_actionable(tmp_path):
    with make_client(tmp_path) as client:
        settings = Settings().model_dump()
        settings["upper"]["camera"] = settings["lower"]["camera"] = None
        assert client.put("/api/settings", json=settings).status_code == 200
        result = client.post("/api/live/start")
        assert result.status_code == 409
        assert "Select" in result.json()["detail"]


def test_compiled_frontend_assets_offline_fonts_and_legacy_removal(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert '<div id="root"></div>' in response.text
        assets = re.findall(r'(?:src|href)="(/static/[^\"]+)"', response.text)
        assert any("/assets/" in path and path.endswith(".js") for path in assets)
        assert "/static/fonts/fonts.css" in assets
        for path in assets:
            assert client.get(path).status_code == 200, path
        fonts = client.get("/static/fonts/fonts.css").text
        for filename in re.findall(r"url\(([^)]+)\)", fonts):
            assert not filename.startswith(("http", "/"))
            assert client.get("/static/fonts/" + filename).status_code == 200
        policy = response.headers["content-security-policy"]
        assert "media-src 'self' blob:" in policy
        assert "'unsafe-inline'" not in policy
        for path in ("app.js", "style.css", "live.css", "tokens.css"):
            assert client.get("/static/" + path).status_code == 404
