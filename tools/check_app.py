"""Real upload/status/playback/download smoke test using the portable backend."""
import argparse
import hashlib
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=ROOT/'configs/models.json')
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    args = parser.parse_args()
    from app.app import create_app
    from app.registry import model_specs
    from app.video import inspect_video
    from tools.model_run import sha256
    specs = model_specs(args.models,args.registry)
    runtime = ROOT/'artifacts'/f'api_smoke_{uuid.uuid4().hex}'
    app = create_app(runtime=runtime,models=args.models,registry=args.registry)
    client = app.test_client()
    jobs = app.extensions['video_jobs']
    try:
        response = client.get('/')
        if response.status_code != 200:
            raise RuntimeError('Frontend did not load')
        page = response.get_data(as_text=True)
        for spec in specs:
            for label in spec.labels:
                if label not in page:
                    raise RuntimeError(f'Frontend missing class {label}')
        if 'Demo model installed' in page or 'does not certify safety compliance' not in page:
            raise RuntimeError('Frontend has obsolete mode or missing limitations')
        token = re.search(r'name="csrf-token" content="([^"]+)"', page)[1]
        with args.source.open('rb') as source:
            response = client.post('/api/jobs', data={'video': (source,args.source.name)},
                                   headers={'X-CSRF-Token':token})
        if response.status_code != 202:
            raise RuntimeError(f'Upload rejected: {response.get_data(as_text=True)}')
        identifier = response.json['id']
        deadline = time.monotonic()+3600
        last_stage = None
        while time.monotonic() < deadline:
            response = client.get(f'/api/jobs/{identifier}')
            if response.status_code != 200:
                raise RuntimeError('Status unavailable')
            job = response.json
            if job['stage'] != last_stage:
                print(f'API job: {job["stage"]}', flush=True)
                last_stage = job['stage']
            if job['state'] == 'failed':
                raise RuntimeError(job['error'])
            if job['state'] == 'complete':
                break
            time.sleep(.5)
        else:
            raise TimeoutError('Video job exceeded one hour')
        jobs.close()
        with client.get(f'/api/jobs/{identifier}/video', headers={'Range':'bytes=0-1023'}) as video:
            if video.status_code != 206 or video.mimetype != 'video/mp4' or len(video.data) != 1024:
                raise RuntimeError('Video byte-range playback failed')
        output = runtime/identifier/'output.mp4'
        with client.get(f'/api/jobs/{identifier}/download') as download:
            if download.status_code != 200 or 'attachment' not in download.headers['Content-Disposition']:
                raise RuntimeError('Download failed')
            if hashlib.sha256(download.data).hexdigest() != sha256(output):
                raise RuntimeError('Downloaded video differs from completed file')
        original, result = inspect_video(args.source), inspect_video(output)
        if (original.frames,original.width,original.height,original.timeline_hash) != (result.frames,result.width,result.height,result.timeline_hash):
            raise RuntimeError('API output did not preserve input frames/timestamps')
        if result.codec != 'h264' or result.audio_streams or abs(result.duration-original.duration) > max(result.time_base,original.time_base):
            raise RuntimeError('Invalid API video format/duration')
        if (runtime/identifier/'input.upload').exists() or any(name in sys.modules for name in ('torch','mmdet','mmcv','tensorrt')):
            raise RuntimeError('Upload cleanup or portable runtime isolation check failed')
        report = (f'PASS: frontend labels {[list(s.labels) for s in specs]}; permanent versioned workflow.\n'
                  f'Upload 202; completed job {identifier}; range playback 206; download 200/hash identical.\n'
                  f'{result.frames} frames; {result.width}x{result.height}; {float(result.fps):.6f} FPS; '
                  f'{float(result.duration):.6f} s; all timestamps preserved; silent H.264.\n'
                  f'No PyTorch imported; input removed; output: {output}\n')
        (runtime/'smoke.txt').write_text(report, encoding='utf-8')
        print(report)
    finally:
        jobs.close()


if __name__ == '__main__':
    main()
