"""Evaluate a selected RTMDet best checkpoint with class-aware COCO metrics.

Development evaluation only. Production inference uses TensorRT.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_run import load_run, identity, category_ids, toml_fields, configure_reference_precision


def match_counts(predictions, ground_truth, threshold=0.30, iou_threshold=0.50):
    """Confidence-ordered, one-to-one matching at a declared operating point."""
    tp = fp = fn = 0
    for image_id in ground_truth.keys() | predictions.keys():
        gt = np.asarray(ground_truth.get(image_id, []), dtype=np.float32).reshape(-1, 4)
        used = set()
        for box, score in sorted(predictions.get(image_id, []), key=lambda p: -p[1]):
            if score < threshold:
                continue
            overlaps = []
            for index, target in enumerate(gt):
                if index in used:
                    continue
                intersection = np.maximum(0, np.minimum(box[2:], target[2:]) - np.maximum(box[:2], target[:2])).prod()
                union = np.maximum(0, box[2:] - box[:2]).prod() + np.maximum(0, target[2:] - target[:2]).prod() - intersection
                overlaps.append((float(intersection / union) if union > 0 else 0, index))
            iou, index = max(overlaps, default=(0, -1))
            if iou >= iou_threshold:
                tp += 1
                used.add(index)
            else:
                fp += 1
        fn += len(gt) - len(used)
    return tp, fp, fn


def draw(frame, instances, labels, threshold=0.30, colors=None):
    colors = colors or [(0,255,0)] * len(labels)
    for box, score, label in zip(instances.bboxes.cpu().numpy(), instances.scores.cpu().numpy(), instances.labels.cpu().numpy()):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        color = colors[int(label)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{labels[int(label)]} {score:.2f}', (max(x1, 0), max(y1-4, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'work_dirs/goggles')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--config', type=Path, help='Resolved release configuration; requires the registered checkpoint hash')
    parser.add_argument('--registry', type=Path, default=ROOT/'configs/models.json')
    parser.add_argument('--video', type=Path, help='Optional qualitative full-video evaluation; never used for fitting')
    args = parser.parse_args()
    os.chdir(ROOT)
    import torch
    from mmdet.apis import inference_detector, init_detector
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    torch.set_num_threads(4)
    configure_reference_precision()
    checkpoint, config_path, config, labels = load_run(args.run_dir, args.checkpoint,args.config)
    from app.registry import read_registry
    entry = next(m for m in read_registry(args.registry)['models'] if m['id'] == args.model_id)
    if tuple(entry['labels']) != labels:
        raise ValueError('Registry labels disagree with saved training configuration')
    from app.registry import sha256
    if sha256(checkpoint) != entry['checkpoint']['sha256']:
        raise ValueError('Checkpoint does not match the registered version')
    provenance = identity(checkpoint, config_path, config, labels)
    model = init_detector(config, str(checkpoint), device='cuda:0')
    output = checkpoint.parent / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    (output/'verified.toml').write_text('status = "running"\n', encoding='utf-8')
    report = [f'Model: {args.model_id}; dataset: {entry["dataset_version"]}', f'Checkpoint: {checkpoint}',
              'Performance depends on training data. This is not a safety certification system.',
              f'Checkpoint SHA256: {provenance["checkpoint_sha256"]}', f'Ordered labels: {labels}',
              'Reference arithmetic: FP32, CUDA matmul and cuDNN TF32 disabled.']
    for split in ('valid', 'test'):
        report_start = len(report)
        dataset = config.val_dataloader.dataset if split == 'valid' else config.test_dataloader.dataset
        annotation_path = Path(dataset.ann_file)
        image_root = Path(dataset.data_prefix.img)
        document = json.loads(annotation_path.read_text())
        ids = category_ids(document, labels)
        gt = {category: defaultdict(list) for category in ids}
        for ann in document['annotations']:
            x, y, w, h = ann['bbox']
            if ann.get('iscrowd', 0) or ann.get('ignore', False):
                raise ValueError('Operating-point metrics require non-crowd, fully annotated data')
            gt[ann['category_id']][ann['image_id']].append(np.array([x, y, x+w, y+h]))
        predictions = {category: defaultdict(list) for category in ids}
        coco_predictions = []
        previews = []
        for image in document['images']:
            path = image_root / image['file_name']
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f'Cannot read {path}')
            with torch.inference_mode():
                result = inference_detector(model, frame).pred_instances
            if not torch.isfinite(result.bboxes).all() or not torch.isfinite(result.scores).all():
                raise RuntimeError('Non-finite checkpoint predictions')
            for box, score, label in zip(result.bboxes.cpu().numpy(), result.scores.cpu().numpy(), result.labels.cpu().numpy()):
                category = ids[int(label)]
                predictions[category][image['id']].append((box, float(score)))
                x1, y1, x2, y2 = box.tolist()
                coco_predictions.append(dict(image_id=image['id'], category_id=category,
                    bbox=[x1, y1, x2-x1, y2-y1], score=float(score)))
            if len(previews) < 3:
                previews.append(cv2.resize(draw(frame, result, labels, colors=entry['colors']), (360, 640)))
        coco = COCO(str(annotation_path))
        if coco_predictions:
            predicted_coco = coco.loadRes(coco_predictions)
        else:
            predicted_coco = COCO()
            predicted_coco.dataset = dict(images=document['images'], categories=document['categories'], annotations=[])
            predicted_coco.createIndex()
        evaluator = COCOeval(coco, predicted_coco, 'bbox')
        evaluator.params.catIds = list(ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        counts = {category: match_counts(predictions[category], gt[category]) for category in ids}
        tp, fp, fn = map(int, np.sum(list(counts.values()), axis=0))
        precision, recall = tp / max(tp+fp, 1), tp / max(tp+fn, 1)
        report.append(f'{split}: precision={precision:.4f}, recall={recall:.4f}, TP={tp}, FP={fp}, FN={fn} '
                      f'(confidence>=0.30, IoU>=0.50); AP50={evaluator.stats[1]:.4f}; mAP50-95={evaluator.stats[0]:.4f}')
        for label, category in zip(labels, ids):
            ctp, cfp, cfn = counts[category]
            k = list(evaluator.params.catIds).index(category)
            precision_grid = evaluator.eval['precision'][:, :, k, 0, -1]
            def mean_valid(values):
                valid = values[values >= 0]
                return float(valid.mean()) if valid.size else float('nan')
            report.append(f'{split}/{label} (COCO ID {category}): precision={ctp/max(ctp+cfp,1):.4f}, '
                          f'recall={ctp/max(ctp+cfn,1):.4f}, TP={ctp}, FP={cfp}, FN={cfn}; '
                          f'AP50={mean_valid(precision_grid[0]):.4f}; mAP50-95={mean_valid(precision_grid):.4f}')
        if not cv2.imwrite(str(output / f'{split}_predictions.jpg'), np.hstack(previews)):
            raise RuntimeError('Cannot save evaluation preview')
        (output/f'{split}_report.txt').write_text('\n'.join(report[:5] + report[report_start:])+'\n', encoding='utf-8')
    if args.video is None:
        report.append('Evaluation completed. Visual review is required before model qualification.')
        (output / 'evaluation.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
        (output/'verified.toml').write_text(toml_fields(dict(status='passed', **provenance)), encoding='utf-8')
        print('\n'.join(report))
        return
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    chosen = set(np.linspace(0, max(0, expected-1), 6, dtype=int))
    index = detections = detected_frames = 0
    previews = []
    start = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            with torch.inference_mode():
                instances = inference_detector(model, frame).pred_instances
            count = int((instances.scores >= 0.30).sum())
            detections += count
            detected_frames += count > 0
            if index in chosen:
                previews.append(cv2.resize(draw(frame, instances, labels), (300, 540)))
            index += 1
            if index % 100 == 0:
                print(f'Evaluated video frame {index}/{expected}', flush=True)
    finally:
        capture.release()
    if index != expected or index == 0:
        raise RuntimeError(f'Video decode incomplete: {index}/{expected}')
    report.append(f'Complete video {args.video.name}: {index} frames, {detections} detections >=0.30, '
                  f'{detected_frames} frames with detections, {time.perf_counter()-start:.1f} seconds.')
    if len(previews) == 6:
        if not cv2.imwrite(str(output / 'video_predictions.jpg'),
                           np.vstack([np.hstack(previews[:3]), np.hstack(previews[3:])])):
            raise RuntimeError('Cannot save factory preview')
    report.append('Evaluation completed. Visual review is required before Phase 3 acceptance.')
    (output / 'evaluation.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (output/'verified.toml').write_text(toml_fields(dict(status='passed', **provenance)), encoding='utf-8')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
