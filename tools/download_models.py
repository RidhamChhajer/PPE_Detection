"""Install hash-verified ONNX assets from a configured release or local URL."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.registry import read_registry, sha256, DEFAULT_REGISTRY, filename


def download(url, target, expected_hash):
    target = Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file() and sha256(target) == expected_hash:
        return
    if urlparse(url).scheme not in ('https','http','file'):
        raise ValueError('Download URL must use https, http (local testing), or file')
    descriptor, temporary = tempfile.mkstemp(prefix=target.name+'.',suffix='.partial',dir=target.parent)
    try:
        with os.fdopen(descriptor,'wb') as output, urlopen(url,timeout=60) as source:
            shutil.copyfileobj(source,output)
        if sha256(temporary) != expected_hash:
            raise ValueError(f'SHA256 mismatch for {target.name}; existing model preserved')
        os.replace(temporary,target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,default=DEFAULT_REGISTRY)
    parser.add_argument('--models',type=Path,default=ROOT/'models')
    parser.add_argument('--base-url',help='Release URL or file:/// local asset directory')
    args = parser.parse_args()
    document = read_registry(args.registry)
    base = args.base_url or document.get('release_base_url')
    if not base:
        parser.error('Release URL is not configured. Set release_base_url in configs/models.json or pass --base-url.')
    for entry in document['models']:
        if entry['enabled']:
            artifact = entry['onnx']
            name = filename(artifact['filename'])
            download(base.rstrip('/')+'/'+name,args.models/name,artifact['sha256'])
            print(f'Verified {entry["id"]}: {args.models/name}')


if __name__ == '__main__': main()
