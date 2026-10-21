"""Build a private Kaggle training ZIP from an explicit allowlist."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'artifacts/kaggle/ppe-rtmdet-l-kaggle.zip'
SOURCES = [
    'configs/goggles_rtmdet_l.py', 'configs/footwear_rtmdet_l.py',
    'tools/train.py', 'tools/stable_optim.py', 'tools/validate_coco.py',
    'tools/validate_training_dataset.py',
    'kaggle/setup_linux.sh', 'kaggle/run_training.py', 'kaggle/README.md',
]


def build():
    files = [ROOT/name for name in SOURCES]
    for task in ('goggles', 'footwear'):
        files.extend(path for path in (ROOT/'data'/task).rglob('*') if path.is_file())
    files = sorted(files)
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f'Invalid package input: {path}')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        manifest = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in files}
        archive.writestr('kaggle/package_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f'ZIP hash mismatch: {name}')
    temporary.replace(OUTPUT)
    print(f'Kaggle training package: {OUTPUT} ({len(files)} files, {OUTPUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    build()
