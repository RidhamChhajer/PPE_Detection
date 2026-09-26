"""Versioned model registry; no training or inference framework imports."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT/'configs/models.json'
CONTRACT = 'rtmdet-bgr-640-pad114-xyxy-sigmoid-v2'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def filename(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Model artifact must be a plain filename')
    return value


@dataclass(frozen=True)
class RegisteredModel:
    entry: dict
    directory: Path
    backend: str

    @property
    def model_id(self): return self.entry['id']
    @property
    def labels(self): return tuple(self.entry['labels'])
    @property
    def colors(self): return tuple(tuple(c) for c in self.entry['colors'])
    @property
    def display_name(self): return self.entry['display_name']
    @property
    def path(self): return self.directory/self.entry[self.backend]['filename']
    @property
    def digest(self): return self.entry[self.backend]['sha256']


def read_registry(path=DEFAULT_REGISTRY):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if document.get('schema_version') != 1 or not isinstance(document.get('models'), list):
        raise ValueError('Unsupported model registry schema')
    ids, files = set(), set()
    for model in document['models']:
        identifier = model.get('id', '')
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*-v[0-9]+', identifier) or identifier in ids:
            raise ValueError('Invalid or duplicate model ID')
        ids.add(identifier)
        if type(model.get('enabled')) is not bool or not model.get('display_name') or not model.get('dataset_version'):
            raise ValueError('Missing model version/display/enable metadata')
        if model.get('version') != identifier.rsplit('-',1)[1]:
            raise ValueError('Model version disagrees with ID')
        labels = model.get('labels')
        if not isinstance(labels,list) or not labels or any(not isinstance(v,str) or not v.strip() for v in labels) or len(labels) != len(set(labels)):
            raise ValueError('Invalid ordered class labels')
        colors = model.get('colors', [])
        if len(colors) != len(labels) or any(len(c) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in c) for c in colors):
            raise ValueError('Each class needs a BGR color')
        if model.get('input_size') != [640,640] or model.get('contract') != CONTRACT:
            raise ValueError('Unsupported input/output contract')
        for key in ('confidence_threshold','nms_iou_threshold'):
            value = model.get(key)
            if type(value) not in (int,float) or not 0 < value <= 1:
                raise ValueError(f'Invalid {key}')
        if model.get('backends') != ['onnx','tensorrt']:
            raise ValueError('Unsupported backend compatibility')
        for kind in ('onnx','checkpoint','tensorrt'):
            artifact = model.get(kind)
            if kind == 'tensorrt' and artifact is None:
                continue
            if not isinstance(artifact,dict):
                raise ValueError(f'Missing {kind} artifact')
            name = filename(artifact.get('filename'))
            if name in files or not re.fullmatch('[0-9a-f]{64}', artifact.get('sha256','')):
                raise ValueError('Duplicate artifact or invalid SHA256')
            files.add(name)
    return document


def model_specs(directory=ROOT/'models', registry=DEFAULT_REGISTRY, backend='onnx', verify=True):
    if backend not in ('onnx','tensorrt'):
        raise ValueError('Backend must be onnx or tensorrt')
    directory = Path(directory).resolve()
    if (directory/'.installing').exists():
        raise RuntimeError('Model installation in progress')
    document = read_registry(registry)
    specifications = []
    for entry in document['models']:
        if not entry['enabled']:
            continue
        if entry.get(backend) is None:
            raise ValueError(f'{entry["id"]}: build the optional TensorRT engine locally first')
        spec = RegisteredModel(entry, directory, backend)
        if verify:
            if not spec.path.is_file():
                raise FileNotFoundError(f'Missing {spec.path.name}; run python tools/download_models.py')
            if sha256(spec.path) != spec.digest:
                raise ValueError(f'Model checksum mismatch: {spec.path.name}')
        specifications.append(spec)
    if not specifications:
        raise ValueError('No enabled detectors in registry')
    if (directory/'.installing').exists():
        raise RuntimeError('Model installation in progress')
    return specifications
