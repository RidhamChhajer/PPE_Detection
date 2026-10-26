from types import SimpleNamespace

import numpy as np
import pytest

from app.config import ViewConfig
from app.inference import Detector, summarize


class StubYOLO:
    device = 0
    names = {0: "goggles", 1: "no_goggles", 2: "empty_wrist", 3: "not_empty_wrist"}

    def predict(self, crop, **kwargs):
        assert kwargs["device"] == 0
        box = SimpleNamespace(cls=np.array(0), conf=np.array(0.9), xyxy=np.array([[1, 2, 11, 12]]))
        return [SimpleNamespace(boxes=[box])]


def test_sahi_wrist_coordinates_follow_roi_and_preserve_goggles(monkeypatch):
    detector = Detector()
    detector.models["upper"] = StubYOLO()
    monkeypatch.setattr(
        detector,
        "sliced_wrists",
        lambda crop, cfg: [
            {"class": "not_empty_wrist", "confidence": 0.8, "box": [10, 20, 20, 30]}
        ],
    )
    detections, _ = detector.predict(
        "upper",
        np.zeros((100, 100, 3), np.uint8),
        ViewConfig(roi=(0.25, 0.25, 0.75, 0.75), sahi_wrist=True),
    )
    assert detections[0]["box"] == [26, 27, 36, 37]
    assert detections[1]["box"] == [35, 45, 45, 55]
    assert summarize(detections) == {
        "goggles": "detected",
        "wrist": "violation",
        "shoes": "unknown",
    }


def test_sahi_off_avoids_slicing_entirely(monkeypatch):
    detector = Detector()
    detector.models["upper"] = StubYOLO()
    monkeypatch.setattr(
        detector,
        "sliced_wrists",
        lambda *_: (_ for _ in ()).throw(AssertionError("SAHI should be off")),
    )
    detections, _ = detector.predict(
        "upper", np.zeros((100, 100, 3), np.uint8), ViewConfig(sahi_wrist=False)
    )
    assert len(detections) == 1


def test_sahi_adapter_filters_non_wrists_and_uses_rgb(monkeypatch):
    detector = Detector()
    detector.models["upper"] = StubYOLO()
    created, calls = [], []

    def wrapper(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(**kwargs)

    def predict(image, model, **kwargs):
        calls.append(kwargs)
        assert image[0, 0].tolist() == [30, 20, 10]
        objects = [
            SimpleNamespace(
                category=SimpleNamespace(name=name),
                score=SimpleNamespace(value=0.8),
                bbox=SimpleNamespace(to_xyxy=lambda: [1, 2, 3, 4]),
            )
            for name in ["goggles", "empty_wrist", "not_empty_wrist"]
        ]
        return SimpleNamespace(object_prediction_list=objects)

    monkeypatch.setattr("sahi.models.ultralytics.UltralyticsDetectionModel", wrapper)
    monkeypatch.setattr("sahi.predict.get_sliced_prediction", predict)
    frame = np.full((100, 100, 3), [10, 20, 30], np.uint8)
    for _ in range(2):
        rows = detector.sliced_wrists(frame, ViewConfig())
        assert [r["class"] for r in rows] == ["empty_wrist", "not_empty_wrist"]
    assert len(created) == 1
    assert created[0]["model"] is detector.models["upper"]
    assert created[0]["device"] == "cuda:0"
    assert calls[0]["perform_standard_pred"] is True
    assert calls[0]["batch_size"] == 2


def test_full_frame_uses_model_device_object():
    detector = Detector()
    device = object()

    def predict(crop, **kwargs):
        assert kwargs["device"] is device
        assert kwargs["quantize"] == 16
        return [SimpleNamespace(boxes=[])]

    detector.models["upper"] = SimpleNamespace(device=device, predict=predict)
    detections, _ = detector.predict(
        "upper", np.zeros((100, 100, 3), np.uint8), ViewConfig(sahi_wrist=False)
    )
    assert detections == []


def test_sahi_rejects_excessive_slice_work(monkeypatch):
    detector = Detector()
    detector.models["upper"] = StubYOLO()
    monkeypatch.setattr(
        "sahi.models.ultralytics.UltralyticsDetectionModel",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    with pytest.raises(ValueError, match="exceeds 16 slices"):
        detector.sliced_wrists(np.zeros((2160, 3840, 3), np.uint8), ViewConfig(sahi_slice=320))
