import threading
import time
from collections import deque

import cv2
import numpy as np

from app.cameras import CameraPool, frame_rate
from app.config import VIEWS
from app.inference import annotate, jpeg, summarize


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
                    fresh = time.monotonic() - result.get("timestamp", 0) < 1
                    detections = (
                        result.get("detections", []) if fresh and not result.get("error") else []
                    )
                try:
                    if detections:
                        image = jpeg(annotate(frame, detections))
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
                try:
                    if view in self.detector.models:
                        detections, ms = self.detector.predict(view, frame, cfg)
                        error = ""
                    state = {
                        "timestamp": stamp,
                        "detections": detections,
                        "inference_ms": ms,
                        "error": error,
                        "latency_ms": round((time.monotonic() - stamp) * 1000, 1),
                    }
                except Exception as exc:
                    state = {
                        "timestamp": stamp,
                        "detections": [],
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
                fresh = running and info["connected"] and age is not None and age < 1000
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
        return {
            "running": running,
            "gpu": dict(self.detector.gpu),
            "views": views,
            "config_warning": self.store.warning,
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
