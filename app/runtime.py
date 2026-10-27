import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from app.cameras import CameraPool, frame_rate
from app.config import ROOT, VIEWS
from app.inference import annotate, associate_ppe_to_persons, jpeg, summarize
from app.audit import audit_store


def placeholder(text):
    frame = np.full((450, 800, 3), (63, 40, 20), np.uint8)
    cv2.putText(
        frame, text, (45, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 230, 240), 1, cv2.LINE_AA
    )
    return jpeg(frame)


class Runtime:
    def __init__(self, store, detector, pool=None):
        self.store, self.detector = store, detector
        self.pool = pool or CameraPool()
        self.lock = threading.RLock()
        self.lifecycle = threading.Lock()
        self.running = False
        self.states = {}
        self.previews = {}
        self.config = store.snapshot()
        self.stop_event = threading.Event()
        self.thread = None
        self.preview_thread = None
        self.generation = 0
        self.empty = placeholder("Camera preview will appear here")

        # Violation snapshot storage & single-pass tracking state
        self.violation_dir = ROOT / "config" / "violations"
        self.violation_dir.mkdir(parents=True, exist_ok=True)
        self.captured_in_pass = {}       # track_id -> timestamp captured during current pass
        self.track_last_seen = {}        # track_id -> timestamp last seen in frame
        self.all_seen_track_ids = set()  # unique worker track IDs observed in session
        self.all_violator_track_ids = set() # unique worker track IDs with violations
        self.workers_current = 0

    def start(self):
        with self.lifecycle:
            if self.running:
                return
            if any(worker and worker.is_alive() for worker in (self.thread, self.preview_thread)):
                raise RuntimeError("Previous camera session is still stopping. Retry shortly.")
            self.config = self.store.snapshot()
            if self.config.upper.camera is None and self.config.lower.camera is None:
                raise ValueError("Select at least one camera in Settings.")
            self.pool.start(self.config)
            self.states = {}
            self.previews = {}
            self.stop_event = threading.Event()
            self.generation += 1
            self.running = True

            # Reset tracking session counters
            with self.lock:
                self.captured_in_pass = {}
                self.track_last_seen = {}
                self.all_seen_track_ids = set()
                self.all_violator_track_ids = set()
                self.workers_current = 0

            self.preview_thread = threading.Thread(
                target=self._preview, args=(self.stop_event,), daemon=True, name="live-preview"
            )
            self.preview_thread.start()
            self.thread = threading.Thread(
                target=self._infer, args=(self.stop_event,), daemon=True, name="live-inference"
            )
            self.thread.start()

    def stop(self):
        with self.lifecycle:
            self.stop_event.set()
            with self.lock:
                self.running = False
                self.states = {}
                self.previews = {}
            if self.preview_thread:
                self.preview_thread.join(timeout=3)
            if self.thread:
                self.thread.join(timeout=15)
                if self.thread.is_alive():
                    raise RuntimeError("GPU operation is still finishing. Retry Stop shortly.")
            released = self.pool.stop()
            with self.lock:
                self.running = False
                self.states = {}
                self.previews = {}
            if not released:
                raise RuntimeError("A USB driver is still releasing a device. Retry Stop shortly.")

    def _preview(self, stop_event):
        seen = {}
        samples = {view: deque() for view in VIEWS}
        while not stop_event.is_set():
            raw = {}
            for view in VIEWS:
                camera = self.pool.devices.get(getattr(self.config, view).camera)
                if camera is None:
                    continue
                frame, stamp, sequence, info = camera.snapshot()
                if frame is None or not info["connected"] or seen.get(view) == sequence:
                    continue
                seen[view] = sequence
                with self.lock:
                    result = self.states.get(view, {})
                    inf_ms = result.get("inference_ms") or 0
                    max_age_sec = max(2.5, (inf_ms * 2.0) / 1000.0)
                    fresh = (time.monotonic() - result.get("timestamp", 0)) < max_age_sec
                    detections = (
                        result.get("detections", []) if fresh and not result.get("error") else []
                    )
                    persons = result.get("persons", []) if fresh and not result.get("error") else []
                try:
                    if detections or persons:
                        image = jpeg(annotate(frame, detections, persons))
                    else:
                        key = (id(camera), sequence)
                        if key not in raw:
                            raw[key] = jpeg(frame)
                        image = raw[key]
                except Exception:
                    continue
                preview = {
                    "jpeg": image,
                    "timestamp": stamp,
                    "sequence": sequence,
                    "fps": frame_rate(samples[view], time.monotonic()),
                }
                with self.lock:
                    if stop_event.is_set():
                        return
                    self.previews[view] = preview
            stop_event.wait(0.005)

    def _infer(self, stop_event):
        seen = {}
        samples = {view: deque() for view in VIEWS}
        while not stop_event.is_set():
            for view in VIEWS:
                if stop_event.is_set():
                    break
                cfg = getattr(self.config, view)
                camera = self.pool.devices.get(cfg.camera)
                if not camera:
                    continue
                frame, stamp, sequence, info = camera.snapshot()
                if frame is None or seen.get(view) == sequence:
                    continue
                seen[view] = sequence
                detections, ms, error = [], None, self.detector.errors.get(view, "")
                persons = []
                enriched_persons = []
                try:
                    # 1. Person tracking with YOLO11n
                    if view == "upper" and getattr(self.detector, "person_model", None) is not None:
                        persons = self.detector.track_persons(frame)
                        now_m = time.monotonic()
                        current_ids = {p["track_id"] for p in persons if p.get("track_id") is not None}
                        with self.lock:
                            self.workers_current = len(persons)
                            for tid in current_ids:
                                self.all_seen_track_ids.add(tid)
                                self.track_last_seen[tid] = now_m

                            # Purge tracks that left the camera footage (absent for >= 3 seconds)
                            # Once purged, if the same worker enters again later, their next pass CAN be captured!
                            expired = [
                                tid for tid, last_t in self.track_last_seen.items()
                                if now_m - last_t >= 3.0
                            ]
                            for tid in expired:
                                self.captured_in_pass.pop(tid, None)
                                self.track_last_seen.pop(tid, None)

                    # 2. PPE Detection
                    if view in self.detector.models:
                        detections, ms = self.detector.predict(view, frame, cfg)
                        error = ""

                    # 3. Associate PPE items to tracked persons
                    if persons:
                        enriched_persons, _ = associate_ppe_to_persons(persons, detections)

                    # 4. Handle Single-Pass Violation Capture
                    now_ts = time.time()
                    dt_now = datetime.fromtimestamp(now_ts)

                    if enriched_persons:
                        for p in enriched_persons:
                            tid = p.get("track_id")
                            if p["is_violator"] and tid is not None:
                                should_capture = False
                                with self.lock:
                                    self.all_violator_track_ids.add(tid)
                                    # Single pass check: capture ONLY ONCE while this worker is in frame
                                    if tid not in self.captured_in_pass:
                                        self.captured_in_pass[tid] = now_ts
                                        should_capture = True

                                if should_capture:
                                    # Save annotated snapshot
                                    annotated = annotate(frame, detections, enriched_persons)
                                    dt_str = dt_now.strftime("%Y-%m-%d %H:%M:%S")
                                    reasons = ", ".join(p.get("violations", []))
                                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 44), (0, 0, 0), -1)
                                    cv2.putText(
                                        annotated,
                                        f"VIOLATION RECORDED | Worker #{tid} ({reasons}) | {dt_str}",
                                        (15, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65,
                                        (0, 0, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )
                                    filename = f"violation_{dt_now.strftime('%Y%m%d_%H%M%S')}_worker_{tid}_{view}.jpg"
                                    filepath = self.violation_dir / filename
                                    try:
                                        cv2.imwrite(str(filepath), annotated)
                                    except Exception as exc_img:
                                        print(f"[Runtime] Could not write snapshot: {exc_img}")

                                    # Record to audit database with worker_id and image_filename
                                    audit_store.record(
                                        view=view,
                                        summary=summarize(detections),
                                        detections=detections,
                                        timestamp=now_ts,
                                        worker_id=tid,
                                        image_filename=filename,
                                    )
                            elif not p["is_violator"] and tid is not None:
                                # Record compliant worker inspection (throttled by audit_store)
                                audit_store.record(
                                    view=view,
                                    summary=summarize(detections),
                                    detections=detections,
                                    timestamp=now_ts,
                                    worker_id=tid,
                                )
                    elif not error and detections is not None:
                        audit_store.record(view, summarize(detections), detections, timestamp=now_ts)

                    state = {
                        "timestamp": time.monotonic(),
                        "detections": detections,
                        "persons": enriched_persons if enriched_persons else persons,
                        "inference_ms": ms,
                        "error": error,
                        "latency_ms": round((time.monotonic() - stamp) * 1000, 1),
                    }
                    # Audit logging handled above for both workers and views
                except Exception as exc:
                    state = {
                        "timestamp": time.monotonic(),
                        "detections": [],
                        "persons": [],
                        "inference_ms": None,
                        "error": f"Inference failed: {exc}",
                        "latency_ms": None,
                    }
                state["processing_fps"] = frame_rate(samples[view], time.monotonic())
                with self.lock:
                    if not stop_event.is_set():
                        self.states[view] = state
            stop_event.wait(0.005)

    def snapshot(self):
        views = {}
        with self.lock:
            running = self.running
            for view in VIEWS:
                cfg = getattr(self.config if running else self.store.snapshot(), view)
                camera = self.pool.devices.get(cfg.camera)
                info = (
                    camera.snapshot()[3]
                    if camera
                    else {
                        "camera": cfg.camera,
                        "connected": False,
                        "message": "Stopped" if not running else "Disabled",
                        "actual": {},
                        "controls": {},
                        "defaults": {},
                    }
                )
                state = self.states.get(view, {})
                preview = self.previews.get(view, {})
                preview_age = (time.monotonic() - preview["timestamp"]) * 1000 if preview else None
                age = (time.monotonic() - state["timestamp"]) * 1000 if state else None
                inf_ms = (state.get("inference_ms") or 0) if state else 0
                max_age_ms = max(2500, inf_ms * 2.0)
                fresh = running and info["connected"] and age is not None and age < max_age_ms
                detections = state.get("detections", []) if fresh and not state.get("error") else []
                views[view] = {
                    **info,
                    "detections": detections,
                    "summary": summarize(detections),
                    "age_ms": round(age) if age is not None else None,
                    "preview_age_ms": round(preview_age)
                    if running and info["connected"] and preview_age is not None
                    else None,
                    "preview_fps": preview.get("fps")
                    if running
                    and info["connected"]
                    and preview_age is not None
                    and preview_age < 1000
                    else None,
                    "inference_ms": state.get("inference_ms") if fresh else None,
                    "processing_fps": state.get("processing_fps") if fresh else None,
                    "latency_ms": state.get("latency_ms") if fresh else None,
                    "model_ready": view in self.detector.models,
                    "error": state.get("error", self.detector.errors.get(view, "")),
                    "stale": running and info["connected"] and not fresh,
                }
            total_workers = len(self.all_seen_track_ids)
            violators = len(self.all_violator_track_ids)
            violation_rate = (
                round((violators / total_workers * 100), 1) if total_workers > 0 else 0.0
            )

        return {
            "running": running,
            "gpu": dict(self.detector.gpu),
            "views": views,
            "config_warning": self.store.warning,
            "workers_current": self.workers_current,
            "workers_total": total_workers,
            "workers_violators": violators,
            "workers_violation_rate": violation_rate,
        }

    def frame(self, view):
        with self.lock:
            cfg = getattr(self.config, view)
            camera = self.pool.devices.get(cfg.camera)
            if not self.running or not camera:
                return self.empty
            if not camera.snapshot()[3]["connected"]:
                return self.empty
            preview = self.previews.get(view)
            if preview and time.monotonic() - preview["timestamp"] < 1:
                return preview["jpeg"]
            return self.empty
