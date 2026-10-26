import io
import json
import time
import zipfile
from collections import Counter
from collections import deque

import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from app.cameras import CameraPool, frame_rate
from app.config import ConfigStore, Settings, ViewConfig
from app.inference import jpeg, summarize, suppress_conflicts
from app.runtime import Runtime
from tools.dataset import choose_splits, prepare, recording, yolo_line


def detection(name, confidence=0.8, box=(0, 0, 20, 20)):
    return {"class": name, "confidence": confidence, "box": box}


def test_visible_violation_overrides_positive_but_absence_is_unknown():
    assert summarize([]) == {"goggles": "unknown", "wrist": "unknown", "shoes": "unknown"}
    rows = [
        detection("goggles"),
        detection("no_goggles", box=(100, 100, 120, 120)),
        detection("empty_wrist"),
    ]
    assert summarize(rows) == {"goggles": "violation", "wrist": "detected", "shoes": "unknown"}
    assert summarize(list(reversed(rows))) == summarize(rows)


def test_conflicting_boxes_suppressed_only_within_category():
    rows = [
        detection("goggles", 0.9),
        detection("no_goggles", 0.7),
        detection("empty_wrist", 0.6),
        detection("no_goggles", 0.8, (30, 0, 50, 20)),
    ]
    kept = suppress_conflicts(rows)
    assert len(kept) == 3
    assert summarize(kept)["goggles"] == "violation"
    assert any(d["class"] == "empty_wrist" for d in kept)


@pytest.mark.parametrize(
    "field,value",
    [
        ("roi", (0, 0.2, 0, 0.5)),
        ("confidence", 1),
        ("imgsz", 961),
        ("camera", -1),
        ("confidence", float("nan")),
    ],
)
def test_config_rejects_invalid_settings(field, value):
    with pytest.raises(ValidationError):
        ViewConfig(**{field: value})


def test_atomic_settings_failure_preserves_previous_value(tmp_path, monkeypatch):
    store = ConfigStore(tmp_path / "settings.json")
    original = Settings()
    store.save(original)
    changed = original.model_copy(deep=True)
    changed.upper.camera = 5
    monkeypatch.setattr("app.config.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        store.save(changed)
    assert store.snapshot() == original
    assert Settings.model_validate_json(store.path.read_text()) == original


class FakeCapture:
    def __init__(self, opened=True):
        self.opened, self.released = opened, False
        self.values = {cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 240}

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.values[key] = value
        return True

    def get(self, key):
        return self.values.get(key, 0)

    def read(self):
        time.sleep(0.01)
        return True, np.zeros((240, 320, 3), np.uint8)

    def release(self):
        self.released = True


def wait_until(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Condition did not become true")


def test_shared_camera_is_opened_once_and_released():
    captures = []

    def factory(index):
        capture = FakeCapture()
        captures.append(capture)
        return capture

    pool = CameraPool(factory)
    settings = Settings(upper=ViewConfig(camera=0), lower=ViewConfig(camera=0))
    pool.start(settings)
    wait_until(lambda: pool.devices[0].snapshot()[0] is not None)
    assert len(captures) == 1
    assert pool.stop()
    assert captures[0].released


def test_failed_camera_does_not_stop_working_camera():
    pool = CameraPool(lambda index: FakeCapture(opened=index == 0))
    pool.start(Settings())
    try:
        wait_until(lambda: pool.devices[0].snapshot()[3]["connected"])
        assert not pool.devices[1].snapshot()[3]["connected"]
    finally:
        pool.stop()


def test_named_device_and_index_alias_share_one_capture(monkeypatch):
    opened = []
    monkeypatch.setattr(
        "app.cameras.camera_catalog", lambda: [{"id": "device-a", "index": 0, "name": "USB camera"}]
    )

    def factory(selection):
        opened.append(selection)
        return FakeCapture()

    monkeypatch.setattr("app.cameras.open_camera", factory)
    pool = CameraPool()
    pool.start(Settings(upper=ViewConfig(camera=0), lower=ViewConfig(camera="device-a")))
    try:
        wait_until(lambda: pool.devices[0].snapshot()[3]["connected"])
        assert opened == ["device-a"]
        assert pool.devices[0] is pool.devices["device-a"]
    finally:
        pool.stop()


class FakeDetector:
    def __init__(self):
        self.models = {"upper": True, "lower": True}
        self.errors = {"upper": "", "lower": ""}
        self.gpu = {"ready": True, "name": "test double", "error": ""}

    def predict(self, view, frame, cfg):
        return [detection("goggles" if view == "upper" else "SAFETY_SHOE")], 2


def test_live_results_clear_after_stop_or_stale(tmp_path):
    runtime = Runtime(
        ConfigStore(tmp_path / "settings.json"), FakeDetector(), CameraPool(lambda _: FakeCapture())
    )
    runtime.start()
    try:
        wait_until(lambda: runtime.snapshot()["views"]["upper"]["summary"]["goggles"] == "detected")
        wait_until(lambda: (runtime.snapshot()["views"]["upper"]["processing_fps"] or 0) > 0)
        with runtime.lock:
            runtime.states["upper"]["timestamp"] = time.monotonic() - 5
            assert runtime.snapshot()["views"]["upper"]["summary"]["goggles"] == "unknown"
            assert runtime.snapshot()["views"]["upper"]["processing_fps"] is None
    finally:
        runtime.stop()
    assert runtime.snapshot()["views"]["upper"]["summary"]["goggles"] == "unknown"
    assert runtime.snapshot()["views"]["upper"]["processing_fps"] is None


def test_measured_frame_rate_uses_recent_intervals_and_resets_after_gap():
    samples = deque()
    assert frame_rate(samples, 0) is None
    assert frame_rate(samples, 0.1) == 10
    assert frame_rate(samples, 0.2) == 10
    assert frame_rate(samples, 2) is None
    assert frame_rate(samples, 2.2) == 5


def test_dataset_grouping_and_yolo_geometry():
    assert recording("i001_WIN_20261023_10_54_15_Pro_mp4-0345.jpg") == "WIN_20261023_10_54_15_Pro"
    assert (
        yolo_line({"class": "goggles", "bbox": [10, 20, 20, 40]}, 100, 100, ["goggles"])
        == "0 0.20000000 0.40000000 0.20000000 0.40000000"
    )
    with pytest.raises(ValueError):
        yolo_line({"class": "goggles", "bbox": [-2, 0, 5, 5]}, 100, 100, ["goggles"])
    records = [{"group": g, "annotations": [{"class": "goggles"}]} for g in ["a", "b", "c", "a"]]
    assert Counter(choose_splits(records).values()) == Counter({"train": 1, "valid": 1, "test": 1})


def test_prepare_preserves_backgrounds_and_remaps_classes(tmp_path):
    archive = tmp_path / "input.zip"
    stream = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(stream, format="PNG")
    spec = {
        "categories": [{"id": 5, "name": "empty_wrist"}, {"id": 3, "name": "SAFETY_SHOE"}],
        "images": [
            {"id": i, "file_name": f"group{i}.png", "width": 100, "height": 100} for i in range(3)
        ],
        "annotations": [{"id": 1, "image_id": 0, "category_id": 5, "bbox": [1, 2, 3, 4]}],
    }
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("train/_annotations.coco.json", json.dumps(spec))
        for i in range(3):
            stream = io.BytesIO()
            Image.new("RGB", (100, 100), (i * 50, 0, 0)).save(stream, format="PNG")
            z.writestr(f"train/group{i}.png", stream.getvalue())
    output = tmp_path / "prepared"
    result = prepare(archive, output, {"group0": "train", "group1": "valid", "group2": "test"})
    assert result["images"] == 3
    assert (output / "upper/labels/train/000000.txt").read_text().startswith("2 ")
    assert (output / "lower/labels/train/000000.txt").read_text() == ""
    with pytest.raises(ValueError):
        prepare(archive, output)


def test_jpeg_round_trip():
    image = np.zeros((40, 80, 3), np.uint8)
    assert (
        cv2.imdecode(np.frombuffer(jpeg(image), dtype=np.uint8), cv2.IMREAD_COLOR).shape
        == image.shape
    )
