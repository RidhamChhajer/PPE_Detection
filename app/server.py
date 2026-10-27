import asyncio
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import ConfigStore, ROOT, Settings, VIEWS
from app.cameras import camera_catalog
from app.inference import Detector
from app.media import MediaTest, detection_views
from app.runtime import Runtime
from app.camera_capabilities import camera_capabilities, validate_controls

MAX_UPLOAD = 250 * 1024 * 1024
ALLOWED = {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"}


def create_app(
    store=None,
    detector=None,
    runtime=None,
    initialize=True,
    scratch_root=None,
    capability_provider=camera_capabilities,
):
    store = store or ConfigStore()
    detector = detector or Detector()
    runtime = runtime or Runtime(store, detector)
    scratch_root = scratch_root or ROOT / "TEMPORARY"
    media = MediaTest(detector, store, scratch_root / "exports")
    uploads = scratch_root / "uploads"
    capabilities = {}

    def query_capabilities(selection):
        result = capability_provider(selection)
        capabilities[str(selection)] = result
        return result

    runtime.pool.capability_provider = query_capabilities

    @asynccontextmanager
    async def lifespan(app):
        uploads.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(uploads)
        for stale in uploads.glob("ppe-*"):
            if stale.is_file():
                stale.unlink(missing_ok=True)
        media.output_dir.mkdir(parents=True, exist_ok=True)
        for stale in media.output_dir.glob("ppe-*.mp4"):
            try:
                stale.unlink(missing_ok=True)
            except PermissionError:
                pass
        scheduler_task = None
        if initialize:
            await asyncio.to_thread(detector.load)
            try:
                from app.audit import audit_store
                from app import reporting
                from app.mailer import daily_scheduler_loop
                scheduler_task = asyncio.create_task(daily_scheduler_loop(audit_store, reporting))
            except Exception as exc:
                print(f"[Server] Could not initialize scheduler: {exc}")
        yield
        if scheduler_task:
            scheduler_task.cancel()
        await asyncio.to_thread(media.stop, True)
        await asyncio.to_thread(runtime.stop)

    app = FastAPI(title="PPE Detection System", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.runtime, app.state.media = runtime, media

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.headers.get("host", "").split(":")[0]
        if host not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse(
                {"detail": "This application accepts local requests only."}, status_code=403
            )
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin != f"http://{request.headers.get('host')}":
                return JSONResponse(
                    {"detail": "Cross-origin changes are not allowed."}, status_code=403
                )
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse(
                    {"detail": "Cross-site changes are not allowed."}, status_code=403
                )
            if request.url.path.startswith("/api/test/"):
                try:
                    size = int(request.headers.get("content-length", "-1"))
                except ValueError:
                    size = -1
                if size < 0:
                    return JSONResponse(
                        {"detail": "Uploads require a known file size."}, status_code=411
                    )
                if size > MAX_UPLOAD + 1024 * 1024:
                    return JSONResponse(
                        {"detail": "File exceeds 250 MB. Use a shorter clip."}, status_code=413
                    )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; font-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
        )
        return response

    @app.get("/api/status")
    def status():
        return runtime.snapshot()

    @app.get("/api/settings")
    def settings():
        return store.snapshot().model_dump()

    @app.get("/api/cameras")
    def cameras():
        return camera_catalog()

    @app.get("/api/camera-capabilities")
    def camera_controls(camera: str):
        with runtime.lifecycle:
            if runtime.running:
                result = capabilities.get(camera)
                if result is None:
                    raise HTTPException(409, "Stop Live to read this camera's control limits.")
                return result
            if media.snapshot()["state"] == "running":
                raise HTTPException(
                    409, "Wait for media processing before refreshing camera controls."
                )
            return query_capabilities(int(camera) if camera.isdigit() else camera)

    @app.put("/api/settings")
    def save_settings(value: Settings):
        with runtime.lifecycle:
            if runtime.running:
                raise HTTPException(409, "Stop Live before saving camera settings.")
            if media.snapshot()["state"] == "running":
                raise HTTPException(409, "Wait for media processing before saving settings.")
            try:
                for key, profile in value.profiles.items():
                    if profile.controls:
                        reported = query_capabilities(int(key) if key.isdigit() else key)
                        validate_controls(profile.model_dump(), reported)
                        profile.device_id = reported["device_id"]
                store.save(value)
            except ValueError as exc:
                raise HTTPException(422, str(exc))
            except OSError:
                raise HTTPException(
                    500, "Settings could not be written. Check project-folder write access."
                )
        return {"message": "Settings saved", "settings": value.model_dump()}

    @app.post("/api/live/{action}")
    def live(action: str):
        try:
            if action == "start":
                runtime.start()
            elif action == "stop":
                runtime.stop()
            else:
                raise HTTPException(404, "Unknown live action")
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc))
        return runtime.snapshot()

    @app.post("/api/models/reload")
    def reload_models():
        if runtime.running or media.snapshot()["state"] == "running":
            raise HTTPException(409, "Stop Live and test playback before reloading models.")
        detector.load()
        return runtime.snapshot()

    @app.get("/api/frame/{view}")
    def frame(view: str):
        if view not in VIEWS:
            raise HTTPException(404, "Unknown camera view")
        with runtime.lock:
            image = runtime.frame(view)
            if image is runtime.empty:
                return Response(status_code=204)
            sequence = runtime.previews[view]["sequence"]
            return Response(
                image,
                media_type="image/jpeg",
                headers={"X-Frame-Sequence": f"{runtime.generation}:{sequence}"},
            )

    @app.post("/api/test/{view}")
    async def test_media(view: str, file: UploadFile):
        if view not in (*VIEWS, "both"):
            raise HTTPException(404, "Unknown test view")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED:
            raise HTTPException(415, "Choose JPEG, PNG, MP4, AVI, MOV or MKV.")
        for selected in detection_views(view):
            if selected not in detector.models:
                raise HTTPException(
                    409, detector.errors.get(selected, f"{selected} model not loaded")
                )
        if media.snapshot()["state"] == "running":
            raise HTTPException(409, "Stop the current test before uploading another file.")
        if shutil.disk_usage(ROOT).free < MAX_UPLOAD * 2:
            raise HTTPException(507, "Not enough free disk space for a temporary media test.")
        uploads.mkdir(parents=True, exist_ok=True)
        path = uploads / f"ppe-{uuid.uuid4().hex}{suffix}"
        try:
            size = 0
            with path.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "File exceeds 250 MB. Use a shorter clip.")
                    handle.write(chunk)
            if not size:
                raise HTTPException(422, "The selected file is empty.")
            try:
                return media.start(path, view)
            except ValueError as exc:
                raise HTTPException(409, str(exc))
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get("/api/test")
    def test_status():
        return media.snapshot()

    @app.get("/api/test/{job_id}/download")
    @app.get("/api/test/{job_id}/video")
    def download_test(job_id: str, request: Request):
        try:
            handle = media.download(job_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(404, str(exc))

        size = handle.seek(0, 2)
        start, end = 0, size - 1
        requested = request.headers.get("range")
        if requested:
            import re

            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            try:
                if not match or not any(match.groups()):
                    raise ValueError()
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), end) if last else end
                else:
                    if int(last) <= 0:
                        raise ValueError()
                    start = max(0, size - int(last))
                if not 0 <= start <= end < size:
                    raise ValueError()
            except ValueError:
                handle.close()
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        handle.seek(start)

        def chunks():
            remaining = end - start + 1
            try:
                while remaining and (chunk := handle.read(min(1024 * 1024, remaining))):
                    remaining -= len(chunk)
                    yield chunk
            finally:
                handle.close()

        disposition = "attachment" if request.url.path.endswith("/download") else "inline"
        headers = {
            "Content-Disposition": f'{disposition}; filename="ppe-detection.mp4"',
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        }
        if requested:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(
            chunks(),
            media_type="video/mp4",
            status_code=206 if requested else 200,
            headers=headers,
        )

    @app.get("/api/test-frame")
    def test_frame():
        with media.lock:
            image = media.image
        return Response(image or runtime.empty, media_type="image/jpeg")

    @app.delete("/api/test")
    def remove_test():
        try:
            media.stop(clear=True)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return media.snapshot()

    @app.get("/api/dashboard/dates")
    def get_dashboard_dates():
        from app.audit import audit_store
        return {"dates": audit_store.get_available_dates()}

    @app.get("/api/dashboard/summary")
    def get_dashboard_summary(date: str = None):
        from app.audit import audit_store
        target_date = date or datetime.now().strftime("%Y-%m-%d")
        return audit_store.get_daily_summary(target_date)

    @app.get("/api/reports/daily/pdf")
    def download_daily_pdf(date: str = None):
        from app.audit import audit_store
        from app.reporting import generate_daily_pdf
        target_date = date or datetime.now().strftime("%Y-%m-%d")
        summary = audit_store.get_daily_summary(target_date)
        pdf_bytes = generate_daily_pdf(target_date, summary)
        return Response(
            pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="ppe-daily-report-{target_date}.pdf"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )

    @app.get("/api/reports/email-config")
    def get_email_config():
        from app.mailer import mailer_service
        return mailer_service.get_config()

    @app.put("/api/reports/email-config")
    async def save_email_config(request: Request):
        from app.mailer import mailer_service
        data = await request.json()
        return mailer_service.update_config(data)

    @app.post("/api/reports/send-email")
    async def trigger_send_email(request: Request):
        from app.audit import audit_store
        from app.reporting import generate_daily_pdf
        from app.mailer import mailer_service
        try:
            body = await request.json()
        except Exception:
            body = {}
        target_date = body.get("date") or datetime.now().strftime("%Y-%m-%d")
        recipients = body.get("recipients")
        summary = audit_store.get_daily_summary(target_date)
        pdf_bytes = generate_daily_pdf(target_date, summary)
        ok, message = mailer_service.send_pdf_email(pdf_bytes, target_date, recipients=recipients)
        if not ok:
            raise HTTPException(400, message)
        return {"success": True, "message": message, "date": target_date}

    @app.get("/api/violations/images/{filename}")
    def get_violation_image(filename: str):
        violation_dir = (ROOT / "config" / "violations").resolve()
        file_path = (violation_dir / filename).resolve()
        if not file_path.is_file() or not str(file_path).startswith(str(violation_dir)):
            raise HTTPException(404, "Violation image snapshot not found")
        return FileResponse(file_path, media_type="image/jpeg")

    @app.get("/api/health")
    def health():
        return {"application": "ppe-guard", "status": "ready"}

    @app.get("/")
    def index():
        return FileResponse(ROOT / "app" / "static" / "index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
    return app


app = create_app()
