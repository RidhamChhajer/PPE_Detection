import io
from pathlib import Path
import re
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.app import create_app, RETENTION_SECONDS
from app.registry import DEFAULT_REGISTRY, LOCAL_REGISTRY
from support import registry_fixture


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        models = self.root/'models'
        models.mkdir()
        registry = registry_fixture(models)
        self.registry = registry
        self.release = threading.Event()
        self.started = threading.Event()
        self.failure = False

        def processor(source, destination, models, progress):
            self.started.set()
            self.release.wait(5)
            if self.failure:
                raise ValueError('Invalid video')
            progress('Processing frames', 1, 1)
            destination.write_bytes(b'completed video')
            result = SimpleNamespace(frames=1, width=64, height=48, fps=25, duration=.04)
            return result, result, 0

        self.app = create_app(self.root/'jobs', models, processor, registry=registry, backend='onnx')
        self.client = self.app.test_client()
        self.jobs = self.app.extensions['video_jobs']
        self.addCleanup(self.jobs.close)
        self.addCleanup(self.release.set)
        page = self.client.get('/').get_data(as_text=True)
        self.assertIn('goggles-v1', page)
        self.assertNotIn('Demo model installed', page)
        self.token = re.search(r'name="csrf-token" content="([^"]+)"', page)[1]

    def upload(self, name='input.mp4', content=b'video', headers=None):
        return self.client.post('/api/jobs', data={'video': (io.BytesIO(content), name)},
                                headers=headers or {'X-CSRF-Token': self.token})

    def finish(self, identifier):
        self.release.set()
        for _ in range(100):
            job = self.jobs.snapshot(identifier)
            if job['state'] in ('complete', 'failed'):
                return job
            time.sleep(.01)
        self.fail('Job did not finish')

    def test_upload_progress_download_and_busy(self):
        response = self.upload()
        self.assertEqual(response.status_code, 202)
        identifier = response.json['id']
        self.assertTrue(self.started.wait(2))
        self.assertEqual(self.client.get(f'/api/jobs/{identifier}/video').status_code, 409)
        self.assertEqual(self.upload().status_code, 409)
        self.assertEqual(self.finish(identifier)['progress'], 100)
        self.assertFalse((self.jobs.root/identifier/'input.upload').exists())
        response = self.client.get(f'/api/jobs/{identifier}/video', headers={'Range': 'bytes=0-3'})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, b'comp')
        response.close()
        response = self.client.get(f'/api/jobs/{identifier}/download')
        self.assertEqual(response.data, b'completed video')
        self.assertIn('attachment', response.headers['Content-Disposition'])
        response.close()

    def test_rejects_invalid_uploads_and_cross_origin(self):
        for name, content in [('bad.exe', b'x'), ('empty.mp4', b'')]:
            self.assertEqual(self.upload(name, content).status_code, 400)
            self.assertIsNone(self.jobs.active)
        self.assertEqual(self.upload(headers={'X-CSRF-Token': 'wrong'}).status_code, 403)
        self.assertEqual(self.upload(headers={'X-CSRF-Token': self.token, 'Origin': 'https://example.com'}).status_code, 403)
        response = self.client.post('/api/jobs', data={'video': [(io.BytesIO(b'a'), 'a.mp4'), (io.BytesIO(b'b'), 'b.mp4')]}, headers={'X-CSRF-Token': self.token})
        self.assertEqual(response.status_code, 400)
        self.app.config['MAX_CONTENT_LENGTH'] = 10
        self.assertEqual(self.upload().status_code, 413)
        self.assertIsNone(self.jobs.active)

    def test_failure_cleans_upload_and_allows_next_job(self):
        self.failure = True
        identifier = self.upload().json['id']
        self.assertEqual(self.finish(identifier)['error'], 'Invalid video')
        self.assertIsNone(self.jobs.active)
        self.assertFalse((self.jobs.root/identifier/'input.upload').exists())
        self.assertEqual(self.client.get(f'/api/jobs/{identifier}/download').status_code, 409)

    def test_expired_outputs_removed_without_touching_unowned_files(self):
        identifier = self.upload().json['id']
        self.finish(identifier)
        unrelated = self.jobs.root/'keep.txt'
        unrelated.write_text('keep')
        self.jobs.jobs[identifier]['finished'] = time.time()-RETENTION_SECONDS-1
        self.assertEqual(self.client.get(f'/api/jobs/{identifier}').status_code, 404)
        self.assertFalse((self.jobs.root/identifier).exists())
        self.assertTrue(unrelated.exists())

    def test_live_routes_and_upload_interlock(self):
        class FakeLive:
            state = 'stopped'
            def status(self):
                return dict(state=self.state, active=self.state == 'running', generation=1,
                            feeds={'face': {'state': self.state}, 'feet': {'state': self.state}})
            def start(self, cameras):
                if cameras != {'face': 0, 'feet': 1}:
                    raise ValueError('Unknown camera')
                self.state = 'running'
            def stop(self):
                self.state = 'stopped'
            def stream(self, role):
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\nimage\r\n'
        fake = FakeLive()
        alternate = create_app(self.root/'live-jobs', self.root/'models',
                               lambda *args: None, registry=self.registry, backend='onnx', live_camera=fake,
                               camera_list=lambda: [{'index': 0, 'name': 'Camera 0'}])
        self.addCleanup(alternate.extensions['video_jobs'].close)
        client = alternate.test_client()
        token = re.search(r'name="csrf-token" content="([^"]+)"', client.get('/').get_data(as_text=True))[1]
        headers = {'X-CSRF-Token': token}
        self.assertEqual(client.get('/api/cameras').json[0]['index'], 0)
        self.assertEqual(client.post('/api/live/start', json={'face': 0, 'feet': 1}).status_code, 403)
        self.assertEqual(client.post('/api/live/start', json=[], headers=headers).status_code, 400)
        self.assertEqual(client.post('/api/live/start', json={'face': 0, 'feet': 0}, headers=headers).status_code, 400)
        self.assertEqual(client.post('/api/live/start', json={'face': 0, 'feet': 1}, headers=headers).status_code, 200)
        self.assertEqual(client.get('/api/live/status').json['state'], 'running')
        response = client.get('/api/live/face/stream')
        self.assertIn(b'Content-Type: image/jpeg', next(response.response))
        response.close()
        self.assertEqual(client.post('/api/jobs', data={'video': (io.BytesIO(b'x'), 'x.mp4')}, headers=headers).status_code, 409)
        self.assertEqual(client.post('/api/live/stop', headers=headers).status_code, 200)
        self.assertEqual(client.get('/api/live/feet/stream').status_code, 409)
        self.assertEqual(client.get('/api/live/other/stream').status_code, 404)

    def test_backend_selects_registry_without_extra_flags(self):
        for backend, registry in [('tensorrt', LOCAL_REGISTRY), ('onnx', DEFAULT_REGISTRY)]:
            with patch('app.runtime.model_specs', return_value=[]) as discover:
                app = create_app(runtime=self.root/backend, backend=backend)
                self.addCleanup(app.extensions['video_jobs'].close)
                self.assertEqual(discover.call_args.args[1:], (registry, backend))


if __name__ == '__main__':
    unittest.main()
