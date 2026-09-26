"""Evaluate-gated portable export, test-set parity, release and local install."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.registry import read_registry, RegisteredModel, DEFAULT_REGISTRY, sha256, CONTRACT
from app.runtime import Detector
from app.detector import preprocess, postprocess
from tools.model_run import load_run, identity, require_evaluation
from tools.export_tensorrt import export, load_wrapper, compare_detections, check_contract, build


def atomic_copy(source,target):
    target = Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent,suffix='.pending')
    os.close(descriptor)
    try:
        shutil.copyfile(source,temporary)
        if sha256(source) != sha256(temporary): raise RuntimeError('Copy checksum mismatch')
        os.replace(temporary,target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(target,document):
    target = Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent,suffix='.pending')
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            json.dump(document,stream,indent=2)
            stream.write('\n')
        os.replace(temporary,target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify(checkpoint,cfg,entry,directory,backend='onnx'):
    import cv2
    import numpy as np
    import torch
    from mmdet.apis import inference_detector
    wrapper, model = load_wrapper(checkpoint,cfg)
    images = json.loads(Path(cfg.test_dataloader.dataset.ann_file).read_text())['images']
    image_root = Path(cfg.test_dataloader.dataset.data_prefix.img)
    if not images: raise ValueError('Empty parity test split')
    maxima = dict(raw_box=0.,raw_score=0.,final_box=0.,final_score=0.)
    counts = np.zeros(len(entry['labels']),dtype=int)
    with Detector(RegisteredModel(entry, directory, backend)) as engine:
        for record in images:
            frame = cv2.imread(str(image_root/record['file_name']))
            tensor, scales = preprocess(frame)
            with torch.inference_mode():
                reference = wrapper(torch.from_numpy(tensor).cuda()).cpu().numpy()
                native = inference_detector(model,frame).pred_instances.cpu()
            actual = engine.infer_raw(tensor)
            if not np.isfinite(reference).all(): raise RuntimeError('Non-finite reference')
            selected = (reference[0,:,4:].max(axis=1)>=.1) | (actual[0,:,4:].max(axis=1)>=.1)
            if selected.any(): maxima['raw_box'] = max(maxima['raw_box'],float(np.abs(reference[0,selected,:4]-actual[0,selected,:4]).max()))
            maxima['raw_score'] = max(maxima['raw_score'],float(np.abs(reference[:,:,4:]-actual[:,:,4:]).max()))
            expected = postprocess(reference,scales,frame.shape)
            native = native[native.scores >= .3]
            native_boxes = native.bboxes.numpy().copy()
            native_boxes[:,[0,2]] = native_boxes[:,[0,2]].clip(0,frame.shape[1])
            native_boxes[:,[1,3]] = native_boxes[:,[1,3]].clip(0,frame.shape[0])
            compare_detections(np.column_stack([native_boxes,native.scores.numpy(),native.labels.numpy()]),expected,.05,.0001)
            detected = postprocess(actual,scales,frame.shape)
            box,score = compare_detections(expected,detected,3.,.03)
            maxima['final_box'],maxima['final_score'] = max(maxima['final_box'],box),max(maxima['final_score'],score)
            counts += np.bincount(detected[:,5].astype(int),minlength=len(counts))
    if maxima['raw_box'] > 2. or maxima['raw_score'] > .03:
        raise RuntimeError(f'Raw parity failed: {maxima}')
    return dict(backend=backend, images=len(images), exact_classes_and_counts=True, detections=dict(zip(entry['labels'],map(int,counts))),maximum_errors=maxima)


def portable_config(cfg,model_id):
    """Resolved release recipe; original saved config remains untouched."""
    cfg = copy.deepcopy(cfg)
    task = model_id.rsplit('-v',1)[0]
    cfg.pop('data_regime',None)
    cfg.model_version = model_id.rsplit('-',1)[1]
    cfg.dataset_version = task+' dataset '+cfg.model_version
    cfg.work_dir = f'work_dirs/{model_id}'
    cfg.load_from = f'{model_id}.pth'
    cfg.resume = False
    for split,name in [('train','train_dataloader'),('valid','val_dataloader'),('test','test_dataloader')]:
        cfg[name].dataset.ann_file = f'data/{task}/annotations/{split}.json'
        cfg[name].dataset.data_prefix.img = f'data/{task}/images/{split}/'
    cfg.val_evaluator.ann_file = cfg.val_dataloader.dataset.ann_file
    cfg.test_evaluator.ann_file = cfg.test_dataloader.dataset.ann_file
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--config',type=Path,help='Resolved release configuration')
    parser.add_argument('--registry',type=Path,default=DEFAULT_REGISTRY)
    parser.add_argument('--release-dir',type=Path,default=ROOT/'release_assets/v1')
    parser.add_argument('--models',type=Path,default=ROOT/'models')
    parser.add_argument('--reuse-onnx',type=Path,help='Reuse a previous export after checker and full parity verification')
    parser.add_argument('--tensorrt-engine',type=Path,help='Verify a locally built engine against this checkpoint')
    parser.add_argument('--tensorrt-identity',help='Embedded identity of a pre-registry local engine')
    parser.add_argument('--build-tensorrt',action='store_true')
    parser.add_argument('--precision',choices=['fp16','stable_scores'],default='fp16')
    args = parser.parse_args()
    os.chdir(ROOT)
    checkpoint,config_path,cfg,labels = load_run(args.run_dir,args.checkpoint,args.config)
    provenance = identity(checkpoint,config_path,cfg,labels)
    require_evaluation(checkpoint.parent,provenance)
    check_contract(cfg)
    document = read_registry(args.registry)
    entry = next(m for m in document['models'] if m['id']==args.model_id)
    if tuple(entry['labels']) != labels or entry['checkpoint']['sha256'] != sha256(checkpoint):
        raise ValueError('Registry checkpoint/class identity mismatch')
    import onnx
    candidates = ROOT/'artifacts/versioned_export'/args.model_id
    candidates.mkdir(parents=True,exist_ok=True)
    onnx_path = candidates/entry['onnx']['filename']
    if args.reuse_onnx: atomic_copy(args.reuse_onnx,onnx_path)
    else: export(checkpoint,cfg,onnx_path)
    graph = onnx.load(str(onnx_path))
    onnx.helper.set_model_props(graph,dict(contract=CONTRACT,model_id=args.model_id,labels=json.dumps(labels)))
    onnx.checker.check_model(graph)
    onnx.save(graph,str(onnx_path))
    entry['onnx']['sha256'] = sha256(onnx_path)
    report = dict(model_id=args.model_id,dataset_version=entry['dataset_version'],checkpoint_filename=checkpoint.name,
                  **provenance,onnx_sha256=entry['onnx']['sha256'])
    report['onnx_parity'] = verify(checkpoint,cfg,entry,candidates)
    if args.build_tensorrt or args.tensorrt_engine:
        engine_path = candidates/f'{args.model_id}.engine'
        if args.build_tensorrt:
            # Build is CPU/TRT-only in a fresh process; no simultaneous model builds.
            import subprocess
            command = [sys.executable,'-c','from pathlib import Path; from tools.export_tensorrt import build; import sys,json; build(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],json.loads(sys.argv[4]),sys.argv[5])',str(onnx_path),str(engine_path),args.model_id,json.dumps(labels),args.precision]
            subprocess.run(command,check=True)
        else: atomic_copy(args.tensorrt_engine,engine_path)
        entry['tensorrt'] = dict(filename=engine_path.name,sha256=sha256(engine_path),identity=args.tensorrt_identity or args.model_id)
        report['tensorrt_parity'] = verify(checkpoint,cfg,entry,candidates,'tensorrt')
    # Every candidate must pass before any existing installed file is replaced.
    args.release_dir.mkdir(parents=True,exist_ok=True)
    atomic_copy(checkpoint,args.release_dir/entry['checkpoint']['filename'])
    portable_config(cfg,args.model_id).dump(str(args.release_dir/f'{args.model_id}_rtmdet_s.py'))
    atomic_copy(onnx_path,args.release_dir/onnx_path.name)
    atomic_copy(onnx_path,args.models/onnx_path.name)
    if entry.get('tensorrt'):
        atomic_copy(candidates/entry['tensorrt']['filename'],args.models/entry['tensorrt']['filename'])
    # Public registry never distributes hardware-specific engine provenance.
    local = copy.deepcopy(document)
    public = copy.deepcopy(document)
    for model in public['models']: model['tensorrt'] = None
    atomic_json(args.registry,public)
    local_path = args.models/'registry.local.json'
    if local_path.exists():
        previous = json.loads(local_path.read_text())
        for model in local['models']:
            if model['id'] != args.model_id:
                prior = next((m for m in previous['models'] if m['id']==model['id']),None)
                if prior: model['tensorrt'] = prior.get('tensorrt')
    atomic_json(local_path,local)
    atomic_json(args.release_dir/'models.json',public)
    atomic_json(args.release_dir/f'{args.model_id}-provenance.json',report)
    atomic_json(checkpoint.parent/'evaluation'/f'{args.model_id}-parity.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
