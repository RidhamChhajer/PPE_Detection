import math
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path

import cv2
import av
from PIL import Image

from app.config import ROOT, VIEWS
from app.inference import annotate, jpeg, summarize

MAX_EXPORT_BYTES = 1024 * 1024 * 1024


def detection_views(view):
    if view == "both":
        return VIEWS
    if view in VIEWS:
        return (view,)
    raise ValueError("Choose upper, lower or both detection views.")


class MediaTest:
    def __init__(self, detector, store, output_dir=None):
        self.detector, self.store = detector, store
        self.output_dir = output_dir or ROOT / "TEMPORARY" / "exports"
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.output = None
        self.job = self.idle_job()
        self.image = None

    @staticmethod
    def idle_job():
        return {
            "id": None,
            "state": "idle",
            "message": "Choose an image or video to test.",
            "download_ready": False,
        }

    def _remove_output(self):
        if self.output:
            try:
                self.output.unlink(missing_ok=True)
            except PermissionError:
                # Windows may hold an active download; startup reaps it after closure.
                pass
            self.output = None

    def start(self, path: Path, view):
        views = detection_views(view)
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("A test is running. Stop it before uploading another file.")
            for selected in views:
                if selected not in self.detector.models:
                    raise ValueError(
                        self.detector.errors.get(selected, f"{selected} model not loaded")
                    )
            self._remove_output()
            self.stop_event = threading.Event()
            self.image = None
            self.job = {
                "id": uuid.uuid4().hex,
                "state": "running",
                "message": "Processing test media",
                "view": view,
                "download_ready": False,
            }
            self.thread = threading.Thread(target=self._run, args=(path, views), daemon=True)
            self.thread.start()
            return self.job.copy()

    def _run(self, path, views):
        capture = writer = None
        output = None
        try:
            settings = self.store.snapshot()
            is_image = path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            fps = 25.0
            if is_image:
                try:
                    with Image.open(path) as header:
                        if header.width * header.height > 40_000_000:
                            raise ValueError(
                                "Image resolution is too large; use at most 40 megapixels."
                            )
                except OSError:
                    raise ValueError("Image could not be decoded. Choose a valid JPEG or PNG.")
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError("Image could not be decoded. Choose a valid JPEG or PNG.")
            else:
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise ValueError("Video could not be decoded. Try an MP4/H.264 or AVI file.")
                if (
                    capture.get(cv2.CAP_PROP_FRAME_WIDTH) * capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    > 40_000_000
                ):
                    raise ValueError("Video resolution is too large; use at most 40 megapixels.")
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps) or not 1 <= fps <= 240:
                    fps = 25.0
            count = 0
            while not self.stop_event.is_set():
                started = time.monotonic()
                if not is_image:
                    ok, image = capture.read()
                    if not ok:
                        if not count:
                            raise ValueError("Video has no decodable frames.")
                        break
                if image.shape[0] * image.shape[1] > 40_000_000:
                    raise ValueError("Media resolution is too large; use at most 40 megapixels.")
                detections = []
                for view in views:
                    if self.stop_event.is_set():
                        break
                    rows, _ = self.detector.predict(view, image, getattr(settings, view))
                    detections.extend(rows)
                if self.stop_event.is_set():
                    break
                ms = round((time.monotonic() - started) * 1000, 1)
                annotated = annotate(image, detections)
                if not is_image:
                    if writer is None:
                        self.output_dir.mkdir(parents=True, exist_ok=True)
                        output = self.output_dir / f"ppe-{self.job['id']}.mp4"
                        h, w = annotated.shape[:2]
                        dimensions = (w - w % 2, h - h % 2)
                        rate = Fraction(str(fps)).limit_denominator(100000)
                        writer = av.open(
                            str(output), mode="w", format="mp4", options={"movflags": "+faststart"}
                        )
                        stream = writer.add_stream("libx264", rate=rate)
                        stream.width, stream.height = dimensions
                        stream.pix_fmt = "yuv420p"
                        stream.options = {"preset": "fast", "crf": "20"}
                    encoded = av.VideoFrame.from_ndarray(
                        cv2.resize(annotated, dimensions), format="bgr24"
                    )
                    encoded.pts, encoded.time_base = count, 1 / rate
                    for packet in stream.encode(encoded):
                        writer.mux(packet)
                    if output.exists() and output.stat().st_size > MAX_EXPORT_BYTES:
                        raise ValueError("Annotated video exceeds 1 GB. Test a shorter clip.")
                count += 1
                with self.lock:
                    self.image = jpeg(annotated)
                    self.job.update(
                        frames=count,
                        inference_ms=ms,
                        summary=summarize(detections),
                        message="Processing video for download"
                        if writer
                        else "Test playback — no live recording",
                    )
                if is_image:
                    break
                self.stop_event.wait(max(0.02, 1 / fps - (time.monotonic() - started)))
            if writer:
                for packet in stream.encode(None):
                    writer.mux(packet)
                writer.close()
                writer = None
                if output.stat().st_size > MAX_EXPORT_BYTES:
                    raise ValueError("Annotated video exceeds 1 GB. Test a shorter clip.")
                check = cv2.VideoCapture(str(output))
                valid = check.isOpened() and check.read()[0]
                check.release()
                if not self.stop_event.is_set() and not valid:
                    raise ValueError(
                        "Encoded video could not be read back. Try a shorter or smaller clip."
                    )
            with self.lock:
                stopped = self.stop_event.is_set()
                if output and not stopped:
                    self.output, output = output, None
                self.job.update(
                    state="stopped" if stopped else "complete",
                    download_ready=self.output is not None,
                    message="Test stopped"
                    if stopped
                    else "Test complete — video ready to download"
                    if self.output
                    else "Test complete — uploaded file removed",
                )
        except Exception as exc:
            with self.lock:
                self.image = None
                self.job.update(state="error", message=str(exc), download_ready=False)
        finally:
            if writer:
                writer.close()
            if capture:
                capture.release()
            if output:
                output.unlink(missing_ok=True)
            path.unlink(missing_ok=True)

    def download(self, job_id):
        with self.lock:
            if self.job["id"] != job_id or not self.job.get("download_ready") or not self.output:
                raise ValueError("No completed detection video is available for this test.")
            return self.output.open("rb")

    def stop(self, clear=False):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=15)
            if self.thread.is_alive():
                raise RuntimeError("Test inference is finishing. Retry Stop shortly.")
        if clear:
            with self.lock:
                self._remove_output()
                self.image = None
                self.job = self.idle_job()

    def snapshot(self):
        with self.lock:
            return dict(self.job, preview_ready=self.image is not None)
