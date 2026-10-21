"""Build and verify TensorRT engines on the recipient PC from a shared bundle."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.registry import model_specs, RegisteredModel, sha256
from app.runtime import Detector
from app.detector import preprocess, postprocess
from tools.export_models import atomic_json
from tools.export_tensorrt import build, compare_detections


def bundle_inputs(root):
    root = Path(root).resolve()
    bundle = json.loads((root/'deployment/bundle.json').read_text(encoding='utf-8'))
    if bundle.get('schema_version') != 1 or sha256(root/'configs/models.json') != bundle.get('registry_sha256'):
        raise ValueError('Bundle registry identity mismatch; extract a complete verified bundle')
    specs = model_specs(root/'models', root/'configs/models.json', 'onnx')
    profiles = bundle.get('models', [])
    if (len(profiles) != len(specs) or len({p['id'] for p in profiles}) != len(profiles)
            or {p['id'] for p in profiles} != {s.model_id for s in specs}):
        raise ValueError('Bundle model list disagrees with the registry')
    inputs = []
    for spec in specs:
        profile = next(p for p in profiles if p['id'] == spec.model_id)
        if profile.get('precision') not in ('fp16', 'fp32') or not profile.get('images'):
            raise ValueError('Each model needs a precision and verification images')
        images = []
        for item in profile['images']:
            path = (root/item['path']).resolve()
            if not path.is_relative_to(root/'verification') or sha256(path) != item['sha256']:
                raise ValueError(f'{spec.model_id}: verification image identity mismatch')
            images.append(path)
        inputs.append((spec, profile['precision'], images))
    return inputs


def verify_engine(reference, candidate, images):
    maxima = dict(raw_box=0.0, raw_score=0.0, final_box=0.0, final_score=0.0)
    with Detector(reference) as cpu, Detector(candidate) as gpu:
        for path in images:
            pixels = cv2.imread(str(path))
            tensor, scales = preprocess(pixels)
            expected, actual = cpu.infer_raw(tensor), gpu.infer_raw(tensor)
            selected = (expected[0, :, 4:].max(axis=1) >= .1) | (actual[0, :, 4:].max(axis=1) >= .1)
            if selected.any():
                maxima['raw_box'] = max(maxima['raw_box'], float(np.abs(expected[0, selected, :4]-actual[0, selected, :4]).max()))
            maxima['raw_score'] = max(maxima['raw_score'], float(np.abs(expected[:, :, 4:]-actual[:, :, 4:]).max()))
            if maxima['raw_box'] > 2.0 or maxima['raw_score'] > .03:
                raise RuntimeError(f'Raw output parity failed on {path.name}: {maxima}')
            entry = reference.entry
            results = [postprocess(raw, scales, pixels.shape, entry['confidence_threshold'], entry['nms_iou_threshold'])
                       for raw in (expected, actual)]
            try:
                box, score = compare_detections(*results, 3.0, .03)
            except RuntimeError as error:
                raise RuntimeError(f'{reference.model_id}: parity failed on {path.name}: {error}') from error
            maxima['final_box'] = max(maxima['final_box'], box)
            maxima['final_score'] = max(maxima['final_score'], score)
    return dict(images=len(images), exact_classes_and_counts=True, maximum_errors=maxima)


def prepare_engines(root=ROOT, builder=build, verifier=verify_engine):
    root = Path(root).resolve()
    inputs = bundle_inputs(root)
    document = copy.deepcopy(json.loads((root/'configs/models.json').read_text(encoding='utf-8')))
    published = []
    report = dict(platform=platform.platform(), python=platform.python_version(),
                  tensorrt=importlib.metadata.version('tensorrt-cu12'), models=[])
    token = uuid.uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix='.build-', dir=root/'models') as temporary:
            directory = Path(temporary)
            for reference, preferred, images in inputs:
                entry = next(e for e in document['models'] if e['id'] == reference.model_id)
                name = f'{reference.model_id}-{token}.engine'
                candidate = directory/name
                precisions = ('fp16', 'fp32') if preferred == 'fp16' else ('fp32',)
                for precision in precisions:
                    print(f'Building {reference.model_id}: {precision} on this PC...', flush=True)
                    builder(reference.path, candidate, reference.model_id, reference.labels, precision)
                    entry['tensorrt'] = dict(filename=name, sha256=sha256(candidate), identity=reference.model_id,
                                             precision=precision)
                    try:
                        parity = verifier(reference, RegisteredModel(entry, directory, 'tensorrt'), images)
                        break
                    except RuntimeError:
                        if precision == 'fp32':
                            raise
                        print(f'{reference.model_id}: FP16 parity failed; rebuilding in FP32.', flush=True)
                report['models'].append(dict(id=reference.model_id, precision=precision, **parity))
            # Unique filenames keep the existing manifest/engines usable until this final commit.
            for entry in document['models']:
                if entry['enabled']:
                    path = root/'models'/entry['tensorrt']['filename']
                    os.replace(directory/path.name, path)
                    published.append(path)
            document['deployment_verification'] = report
            atomic_json(root/'models/registry.local.json', document)
            published.clear()
    finally:
        for path in published:
            path.unlink(missing_ok=True)
    print('PASS: both target-PC engines verified and registry installed.', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true', help='Verify included CPU models without building GPU engines')
    parser.add_argument('--backend', choices=['auto', 'openvino', 'onnx'],
                        help='Verify an installed runtime without building TensorRT engines')
    args = parser.parse_args()
    if args.cpu or args.backend:
        from app.runtime import select_runtime
        inputs = bundle_inputs(ROOT)
        specs, _, selected = select_runtime(ROOT/'models', backend='onnx' if args.cpu else args.backend)
        print(f"Inference: {selected['backend']} / {selected['device']}", flush=True)
        for reason in selected['fallback_reasons']:
            print(f'Auto skipped {reason}', flush=True)
        images_by_id = {spec.model_id: images for spec, _, images in inputs}
        for spec in specs:
            with Detector(spec) as detector:
                detector.detect(cv2.imread(str(images_by_id[spec.model_id][0])))
            print(f"{selected['backend']} PASS: {spec.model_id}", flush=True)
    else:
        prepare_engines()


if __name__ == '__main__':
    main()
