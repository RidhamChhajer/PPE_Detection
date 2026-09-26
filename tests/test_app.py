import hashlib
import io
from pathlib import Path
import re
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from app.app import create_app, RETENTION_SECONDS
from support import registry_fixture


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        models = self.root/'models'
        models.mkdir()
        registry = registry_fixture(models)
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

        self.app = create_app(self.root/'jobs', models, processor, registry=registry)
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


if __name__ == '__main__':
    unittest.main()
