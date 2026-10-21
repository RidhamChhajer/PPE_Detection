"""Create a code+models Windows bundle; never copy the development GPU engines."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.registry import read_registry, sha256
from tools.prepare_repository import source_files, check_links, findings


def deployment_assets(root=ROOT):
    root = Path(root).resolve()
    document = read_registry(root/'configs/models.json')
    assets, profiles = {}, []
    for entry in document['models']:
        if not entry['enabled']:
            continue
        for kind, folder in [('onnx', 'models'), ('checkpoint', 'checkpoints')]:
            artifact = entry[kind]
            source = (root/'models'/artifact['filename']) if kind == 'onnx' else (root/'release_assets/v2'/artifact['filename'])
            if sha256(source) != artifact['sha256']:
                raise ValueError(f'Model hash mismatch: {source.name}')
            assets[f'{folder}/{artifact["filename"]}'] = source
        config = root/'release_assets/v2'/f'{entry["id"]}_rtmdet.py'
        if not config.is_file():
            raise FileNotFoundError(f'Missing selected checkpoint configuration: {config.name}')
        assets[f'checkpoints/{config.name}'] = config
        task = entry['id'].rsplit('-v', 1)[0]
        annotation = json.loads((root/'data'/task/'annotations/test.json').read_text(encoding='utf-8'))
        profile = dict(id=entry['id'], precision='fp32' if entry['camera_role'] == 'feet' else 'fp16', images=[])
        for index, image in enumerate(annotation['images']):
            source = (root/'data'/task/'images/test'/image['file_name']).resolve()
            if not source.is_relative_to(root/'data'/task/'images/test') or not source.is_file():
                raise ValueError('Unsafe or missing verification image')
            name = f'verification/{entry["id"]}/{index:04d}{source.suffix.lower()}'
            assets[name] = source
            profile['images'].append(dict(path=name, sha256=sha256(source)))
        if not profile['images']:
            raise ValueError(f'{entry["id"]}: no engine verification images')
        profiles.append(profile)
    if not profiles:
        raise ValueError('No enabled models')
    return assets, dict(schema_version=1, registry_sha256=sha256(root/'configs/models.json'), models=profiles)


def package(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new deployment ZIP; existing deliveries are not overwritten')
    sources = source_files()
    check_links(sources)
    for path in sources:
        if any(item['kind'] == 'potential_secret' for item in findings(path.read_text(encoding='utf-8'), path.name)):
            raise ValueError(f'Source may contain a credential: {path.name}')
    files = {path.relative_to(ROOT).as_posix(): path for path in sources}
    assets, bundle = deployment_assets()
    files.update(assets)
    generated = json.dumps(bundle, indent=2).encode('utf-8')
    hashes = {name: sha256(path) for name, path in files.items()}
    hashes['deployment/bundle.json'] = hashlib.sha256(generated).hexdigest()
    sums = ''.join(f'{digest}  {name}\n' for name, digest in sorted(hashes.items())).encode('utf-8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial.zip')
    if temporary.exists():
        raise FileExistsError('A partial deployment archive already exists')
    prefix = 'PPE_Detection/'
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, path in sorted(files.items()):
                compression = zipfile.ZIP_STORED if path.suffix in ('.onnx', '.pth') else zipfile.ZIP_DEFLATED
                archive.write(path, prefix+name, compress_type=compression)
            archive.writestr(prefix+'deployment/bundle.json', generated)
            archive.writestr(prefix+'BUNDLE_SHA256SUMS.txt', sums)
        with zipfile.ZipFile(temporary) as archive:
            if set(archive.namelist()) != {prefix+name for name in hashes} | {prefix+'BUNDLE_SHA256SUMS.txt'}:
                raise RuntimeError('Deployment archive inventory mismatch')
            for name, digest in hashes.items():
                with archive.open(prefix+name) as stream:
                    actual = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024*1024), b''):
                        actual.update(block)
                if actual.hexdigest() != digest:
                    raise RuntimeError(f'Deployment archive hash mismatch: {name}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'PASS: {destination} ({destination.stat().st_size:,} bytes)', flush=True)
    print(f'{len(files)} files plus manifest/checksums; no dependencies or development engines.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    package(parser.parse_args().destination)
