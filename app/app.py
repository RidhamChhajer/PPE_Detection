"""Local, single-job PPE video service. Start with: python -m app.app"""
import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import secrets
import threading
import time
import uuid
from functools import partial

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from .registry import model_specs, DEFAULT_REGISTRY
from .video import process_video

ROOT = Path(__file__).resolve().parents[1]
RETENTION_SECONDS = 24 * 60 * 60
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}


class VideoJobs:
    def __init__(self, runtime, models, logger, processor=process_video):
        self.root = Path(runtime).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.models = Path(models).resolve()
        self.logger = logger
        self.processor = processor
        self.lock = threading.Lock()
        self.jobs = {}
        self.active = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ppe-video')
        self.cleanup()

    def cleanup(self):
        """Remove only expired, application-owned files in UUID job directories."""
        cutoff = time.time() - RETENTION_SECONDS
        with self.lock:
            for directory in self.root.iterdir():
                if not directory.is_dir() or not re.fullmatch(r'[0-9a-f]{32}', directory.name):
                    continue
                if directory.resolve().parent != self.root or directory.name == self.active:
                    continue
                job = self.jobs.get(directory.name)
                modified = job.get('finished', job['created']) if job else directory.stat().st_mtime
                if modified >= cutoff:
                    continue
                for filename in ('input.upload', 'output.mp4', 'output.partial.mp4'):
                    path = directory / filename
                    if path.resolve().parent == directory.resolve():
                        try:
                            path.unlink(missing_ok=True)
                        except PermissionError:
                            self.logger.info('Retaining an output currently in use')
                try:
                    directory.rmdir()
                except OSError:
                    continue
                self.jobs.pop(directory.name, None)

    def reserve(self):
        self.cleanup()
        with self.lock:
            if self.active is not None:
                raise RuntimeError('A video is already being processed. Wait for it to finish.')
            identifier = uuid.uuid4().hex
            directory = self.root / identifier
            directory.mkdir()
            self.jobs[identifier] = dict(id=identifier, state='uploading', stage='Uploading', progress=0,
                                         created=time.time(), filename='', error=None)
            self.active = identifier
            return identifier, directory

    def fail(self, identifier, message):
        with self.lock:
            self.jobs[identifier].update(state='failed', stage='Failed', error=message, finished=time.time())
            if self.active == identifier:
                self.active = None
        (self.root / identifier / 'input.upload').unlink(missing_ok=True)

    def start(self, identifier, filename):
        with self.lock:
            self.jobs[identifier].update(filename=filename, state='processing')
        self.executor.submit(self._process, identifier)

    def _process(self, identifier):
        directory = self.root / identifier

        def progress(stage, count, total):
            fraction = min(count / max(total, 1), 1)
            start, span = {'Inspecting input': (0, 10), 'Processing frames': (10, 80),
                           'Verifying output': (90, 9), 'Complete': (99, 0)}[stage]
            with self.lock:
                self.jobs[identifier].update(stage=stage, progress=round(start+span*fraction),
                                             frames_done=count, frames_total=total)
        try:
            _, result, _ = self.processor(directory/'input.upload', directory/'output.mp4', self.models, progress)
            with self.lock:
                self.jobs[identifier].update(state='complete', stage='Complete', progress=100,
                    finished=time.time(), frames=result.frames, width=result.width, height=result.height,
                    fps=float(result.fps), duration=float(result.duration))
                self.active = None
        except Exception as error:
            self.logger.exception('Video job %s failed', identifier)
            message = str(error) if isinstance(error, ValueError) else 'Processing failed. Check the local server log for details.'
            self.fail(identifier, message)
        finally:
            (directory/'input.upload').unlink(missing_ok=True)

    def snapshot(self, identifier):
        self.cleanup()
        with self.lock:
            job = self.jobs.get(identifier)
            return dict(job) if job else None

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=False)


def create_app(runtime=None, models=None, processor=None, registry=DEFAULT_REGISTRY, backend='onnx', provider='cpu'):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES, MAX_FORM_PARTS=10,
                      TRUSTED_HOSTS=['127.0.0.1', 'localhost'])
    models = Path(models or ROOT/'models')
    specs = model_specs(models, registry, backend)
    processor = processor or partial(process_video, registry=registry, backend=backend, provider=provider)
    token = secrets.token_urlsafe(32)
    jobs = VideoJobs(runtime or ROOT/'artifacts/app', models, app.logger, processor)
    app.extensions['video_jobs'] = jobs
    atexit.register(jobs.close)

    @app.before_request
    def local_request():
        if request.method == 'POST':
            origin = request.headers.get('Origin')
            if origin and origin != request.host_url.rstrip('/'):
                abort(403, 'Cross-origin uploads are not allowed')
            if not secrets.compare_digest(request.headers.get('X-CSRF-Token', ''), token):
                abort(403, 'Reload the page before uploading')

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if response.mimetype != 'video/mp4':
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        description = 'Video exceeds the 512 MB upload limit.' if error.code == 413 else error.description
        return jsonify(error=description), error.code

    @app.get('/')
    def index():
        return render_template('index.html', token=token, specs=specs)

    @app.post('/api/jobs')
    def upload():
        try:
            identifier, directory = jobs.reserve()
        except RuntimeError as error:
            abort(409, str(error))
        try:
            if len(request.files.getlist('video')) != 1 or set(request.files) != {'video'}:
                abort(400, 'Choose exactly one video to upload')
            uploaded = request.files.get('video')
            if uploaded is None or not uploaded.filename:
                abort(400, 'Choose one video to upload')
            filename = secure_filename(uploaded.filename)
            if Path(filename).suffix.lower() not in EXTENSIONS:
                abort(400, 'Choose an MP4, MOV, MKV, AVI, WebM, or M4V video')
            uploaded.save(directory/'input.upload')
            if (directory/'input.upload').stat().st_size == 0:
                abort(400, 'The uploaded video is empty')
            jobs.start(identifier, filename)
        except Exception as error:
            jobs.fail(identifier, str(error))
            raise
        return jsonify(id=identifier), 202

    @app.get('/api/jobs/<identifier>')
    def status(identifier):
        job = jobs.snapshot(identifier)
        if job is None:
            abort(404, 'Job not found; it may have expired or the server may have restarted')
        return jsonify(job)

    def output_path(identifier):
        job = jobs.snapshot(identifier)
        if job is None:
            abort(404)
        if job['state'] != 'complete':
            abort(409, 'The complete output is not ready yet')
        path = jobs.root / identifier / 'output.mp4'
        if not path.is_file():
            abort(404, 'Output expired')
        return path

    @app.get('/api/jobs/<identifier>/video')
    def video(identifier):
        return send_file(output_path(identifier), mimetype='video/mp4', conditional=True)

    @app.get('/api/jobs/<identifier>/download')
    def download(identifier):
        return send_file(output_path(identifier), mimetype='video/mp4', as_attachment=True,
                         download_name='ppe_annotated.mp4', conditional=True)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--backend', choices=['onnx','tensorrt'], default='onnx')
    parser.add_argument('--provider', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a port between 1024 and 65535')
    from waitress import serve
    app = create_app(models=args.models, registry=args.registry, backend=args.backend, provider=args.provider)
    print(f'PPE Video Review: http://127.0.0.1:{args.port}', flush=True)
    print('Current model status is displayed on the page. Ctrl+C stops the server.', flush=True)
    try:
        serve(app, host='127.0.0.1', port=args.port, threads=4, connection_limit=32,
              max_request_body_size=MAX_UPLOAD_BYTES, expose_tracebacks=False)
    finally:
        app.extensions['video_jobs'].close()


if __name__ == '__main__':
    main()
