import os
import threading
import time

import cv2

from app.config import CLASSES, ROOT

# Keep Ultralytics diagnostics local; inference must never fetch assets or emit telemetry.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "config" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

CATEGORY = {
    "goggles": "goggles",
    "no_goggles": "goggles",
    "empty_wrist": "wrist",
    "not_empty_wrist": "wrist",
    "SAFETY_SHOE": "shoes",
    "NOT_SAFETY_SHOE": "shoes",
}
BAD = {"no_goggles", "not_empty_wrist", "NOT_SAFETY_SHOE"}
LABELS = {
    "goggles": "Safety goggles",
    "no_goggles": "No safety goggles",
    "empty_wrist": "Clear wrist",
    "not_empty_wrist": "Wrist accessory",
    "SAFETY_SHOE": "Approved shoe",
    "NOT_SAFETY_SHOE": "Other shoe",
}


def overlap(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def suppress_conflicts(detections, threshold=0.55):
    kept = []
    for item in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        if not any(
            CATEGORY[item["class"]] == CATEGORY[k["class"]]
            and overlap(item["box"], k["box"]) > threshold
            for k in kept
        ):
            kept.append(item)
    return kept


def summarize(detections):
    result = {k: "unknown" for k in ("goggles", "wrist", "shoes")}
    for item in detections:
        key = CATEGORY[item["class"]]
        if item["class"] in BAD:
            result[key] = "violation"
        elif result[key] != "violation":
            result[key] = "detected"
    return result


def associate_ppe_to_persons(persons, detections):
    """Associate PPE detections to person bounding boxes."""
    enriched_persons = []
    for p in persons:
        enriched_persons.append({
            "track_id": p.get("track_id"),
            "box": p["box"],
            "confidence": p.get("confidence", 1.0),
            "ppe_items": [],
            "violations": [],
            "compliant": [],
            "is_violator": False,
        })

    unassigned_ppe = []
    for d in detections:
        box = d["box"]
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0

        matched_person = None
        best_dist = float("inf")
        for p in enriched_persons:
            px1, py1, px2, py2 = p["box"]
            pad_x = (px2 - px1) * 0.12
            pad_y = (py2 - py1) * 0.12
            if (px1 - pad_x) <= cx <= (px2 + pad_x) and (py1 - pad_y) <= cy <= (py2 + pad_y):
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                dist = (cx - pcx) ** 2 + (cy - pcy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    matched_person = p

        if matched_person:
            matched_person["ppe_items"].append(d)
            if d["class"] in BAD:
                matched_person["violations"].append(d["class"])
                matched_person["is_violator"] = True
            else:
                matched_person["compliant"].append(d["class"])
        else:
            unassigned_ppe.append(d)

    return enriched_persons, unassigned_ppe


def annotate(frame, detections, persons=None):
    image = frame.copy()
    scale = max(0.5, image.shape[1] / 1900)

    # 1. Annotate Person Bounding Boxes
    if persons:
        for p in persons:
            is_violator = p.get("is_violator", False)
            tid = p.get("track_id")
            # Red for violator, Green for compliant
            p_color = (0, 0, 220) if is_violator else (34, 197, 94)
            x1, y1, x2, y2 = map(int, p["box"])
            cv2.rectangle(image, (x1, y1), (x2, y2), p_color, max(2, int(scale * 3)))

            tid_str = f"Worker #{tid}" if tid is not None else "Worker"
            if is_violator:
                reasons = p.get("violations", [])
                reason_labels = [LABELS.get(r, r) for r in reasons]
                reason_str = ", ".join(reason_labels) if reason_labels else "Violation"
                badge = f"VIOLATION: {tid_str} ({reason_str})"
            else:
                badge = f"COMPLIANT: {tid_str}"

            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.65, 2)
            tx = max(0, min(x1, image.shape[1] - tw - 12))
            ty = max(th + 12, min(y1, image.shape[0] - 1))
            cv2.rectangle(image, (tx, ty - th - 10), (tx + tw + 10, ty), p_color, -1)
            cv2.putText(
                image,
                badge,
                (tx + 5, ty - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale * 0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # 2. Annotate PPE Detections
    for d in detections:
        color = (71, 53, 188) if d["class"] in BAD else (100, 123, 8)
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, int(scale * 2)))
        text = f"{LABELS.get(d['class'], d['class'])} {d['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, 1)
        tx = max(0, min(x1, image.shape[1] - tw - 8))
        ty = max(th + 10, min(y1, image.shape[0] - 1))
        cv2.rectangle(image, (tx, ty - th - 8), (tx + tw + 8, ty), color, -1)
        cv2.putText(
            image,
            text,
            (tx + 4, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def jpeg(frame, max_width=1280):
    if frame.shape[1] > max_width:
        frame = cv2.resize(frame, (max_width, round(frame.shape[0] * max_width / frame.shape[1])))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Frame could not be encoded")
    return encoded.tobytes()


class Detector:
    def __init__(self):
        self.lock = threading.RLock()
        self.models = {}
        self.person_model = None
        self.wrist_slicer = None
        self.errors = {}
        self.device = "cuda:0"
        self.gpu = {"ready": False, "name": "CUDA not checked", "error": "Run Setup GPU.cmd"}

    def load(self):
        with self.lock:
            self.wrist_slicer = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.ones(1, device="cuda:0").add_(1)
                    torch.cuda.synchronize()
                    self.device = "cuda:0"
                    self.gpu = {"ready": True, "name": torch.cuda.get_device_name(0), "error": ""}
                else:
                    self.device = "cpu"
                    self.gpu = {"ready": True, "name": "CPU", "error": ""}
            except ImportError:
                self.gpu = {
                    "ready": False,
                    "name": "GPU setup required",
                    "error": "GPU runtime not installed. Run Setup GPU.cmd.",
                }
            except Exception as exc:
                self.gpu = {"ready": False, "name": "GPU unavailable", "error": str(exc)}

            # Load YOLO11n for Person Detection & Tracking
            try:
                from ultralytics import YOLO
                person_model_path = ROOT / "models" / "yolo11n.pt"
                src = str(person_model_path) if person_model_path.is_file() else "yolo11n.pt"
                self.person_model = YOLO(src)
                self.person_model.to(self.device)
            except Exception as exc:
                print(f"[Detector] Could not load person tracking model (yolo11n): {exc}")
                self.person_model = None

            for view in CLASSES:
                self.models.pop(view, None)
                path = ROOT / "models" / f"{view}.pt"
                if not path.is_file():
                    self.errors[view] = (
                        f"Custom model missing: models/{view}.pt. Train and install the PPE weights."
                    )
                    continue
                if not self.gpu["ready"]:
                    self.errors[view] = self.gpu["error"]
                    continue
                try:
                    from ultralytics import YOLO, settings

                    settings.update({"sync": False})

                    model = YOLO(str(path), task="detect")
                    if set(model.names.values()) != set(CLASSES[view]):
                        raise ValueError(
                            f"Model classes must be exactly {', '.join(CLASSES[view])}"
                        )
                    model.to(self.device)
                    # SAHI must inherit the same precision as the full-frame pass to reuse its backend.
                    if self.device.startswith("cuda"):
                        model.overrides["quantize"] = 16
                    self.models[view] = model
                    self.errors[view] = ""
                except Exception as exc:
                    self.errors[view] = f"Cannot load {view} PPE model: {exc}"

    def track_persons(self, frame, conf=0.45):
        """Detect and track persons with persistent IDs across frames."""
        with self.lock:
            if self.person_model is None:
                return []
            try:
                results = self.person_model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    tracker="bytetrack.yaml",
                    conf=conf,
                    verbose=False,
                    device=self.device,
                )
                persons = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().tolist()
                    confs = results[0].boxes.conf.cpu().tolist()
                    track_ids = (
                        results[0].boxes.id.int().cpu().tolist()
                        if results[0].boxes.id is not None
                        else [None] * len(boxes)
                    )
                    for box, tid, score in zip(boxes, track_ids, confs):
                        persons.append({
                            "track_id": int(tid) if tid is not None else None,
                            "box": [round(c, 1) for c in box],
                            "confidence": float(score),
                        })
                return persons
            except Exception as exc:
                print(f"[Detector] Person tracking error: {exc}")
                return []

    def sliced_wrists(self, crop, cfg):
        from sahi.models.ultralytics import UltralyticsDetectionModel
        from sahi.predict import get_sliced_prediction

        if self.wrist_slicer is None:
            self.wrist_slicer = UltralyticsDetectionModel(
                model=self.models["upper"],
                device=self.device,
                confidence_threshold=cfg.confidence,
                image_size=cfg.sahi_slice,
            )
        self.wrist_slicer.confidence_threshold = cfg.confidence
        self.wrist_slicer.image_size = cfg.sahi_slice
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # Limit the slice count by rejecting oversized regions, never by silently dropping wrists.
        from sahi.slicing import get_slice_bboxes

        slices = get_slice_bboxes(
            image_height=crop.shape[0],
            image_width=crop.shape[1],
            slice_height=cfg.sahi_slice,
            slice_width=cfg.sahi_slice,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            auto_slice_resolution=False,
        )
        if len(slices) > 16:
            raise ValueError(
                "Wrist SAHI region exceeds 16 slices. Increase slice size or narrow the upper detection region in Settings."
            )
        result = get_sliced_prediction(
            rgb,
            self.wrist_slicer,
            slice_height=cfg.sahi_slice,
            slice_width=cfg.sahi_slice,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            # The slice-scale full view recovers close wrists that lose context in tiles.
            perform_standard_pred=True,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=False,
            exclude_classes_by_name=["goggles", "no_goggles"],
            batch_size=2,
            verbose=0,
        )
        return [
            {
                "class": obj.category.name,
                "confidence": float(obj.score.value),
                "box": obj.bbox.to_xyxy(),
            }
            for obj in result.object_prediction_list
            if obj.category.name in {"empty_wrist", "not_empty_wrist"}
        ]

    def predict(self, view, frame, cfg):
        with self.lock:
            if view not in self.models:
                raise RuntimeError(self.errors.get(view, "Model not loaded"))
            h, w = frame.shape[:2]
            left, top, right, bottom = cfg.roi
            x, y = int(left * w), int(top * h)
            crop = frame[y : max(y + 1, int(bottom * h)), x : max(x + 1, int(right * w))]
            start = time.monotonic()
            results = self.models[view].predict(
                crop,
                device=self.models[view].device,
                imgsz=cfg.imgsz,
                conf=cfg.confidence,
                verbose=False,
                quantize=16,
                save=False,
                max_det=100,
            )
            detections = []
            for box in results[0].boxes:
                name = self.models[view].names[int(box.cls.item())]
                if name not in CLASSES[view]:
                    continue
                a, b, c, d = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class": name,
                        "confidence": float(box.conf.item()),
                        "box": [a + x, b + y, c + x, d + y],
                    }
                )
            if view == "upper" and cfg.sahi_wrist:
                wrist_predictions = self.sliced_wrists(crop, cfg)
                for item in wrist_predictions:
                    a, b, c, d = item["box"]
                    item["box"] = [a + x, b + y, c + x, d + y]
                # Keep full-frame wrists too: a hand straddling a slice boundary can lose context.
                detections.extend(wrist_predictions)
            return suppress_conflicts(detections), round((time.monotonic() - start) * 1000, 1)
