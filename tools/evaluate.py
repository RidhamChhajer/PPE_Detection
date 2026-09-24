"""Evaluate the selected goggles checkpoint and inspect the complete factory clip.

Development evaluation only. Production inference uses TensorRT.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def best_checkpoint(run_dir=None):
    candidates = list((run_dir or ROOT / 'work_dirs/goggles').glob('best_*.pth'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one selected best checkpoint, found {len(candidates)}')
    return candidates[0]


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


def draw(frame, instances, threshold=0.30):
    for box, score in zip(instances.bboxes.cpu().numpy(), instances.scores.cpu().numpy()):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'goggles {score:.2f}', (max(x1, 0), max(y1-4, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'work_dirs/goggles')
    parser.add_argument('--video', type=Path, help='Optional qualitative full-video evaluation; never used for fitting')
    args = parser.parse_args()
    os.chdir(ROOT)
    import torch
    from mmdet.apis import inference_detector, init_detector
    from mmengine.config import Config
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    torch.set_num_threads(4)
    checkpoint = (args.checkpoint or best_checkpoint(args.run_dir)).resolve()
    config = Config.fromfile(str(checkpoint.parent / 'goggles_rtmdet_s.py'))
    model = init_detector(config, str(checkpoint), device='cuda:0')
    output = checkpoint.parent / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    report = [f'Checkpoint: {checkpoint}', f'Data regime: {config.get("data_regime", "sample")}. '
              'Metrics do not by themselves establish industrial qualification.']
    for split in ('valid', 'test'):
        dataset = config.val_dataloader.dataset if split == 'valid' else config.test_dataloader.dataset
        annotation_path = Path(dataset.ann_file)
        image_root = Path(dataset.data_prefix.img)
        document = json.loads(annotation_path.read_text())
        gt = defaultdict(list)
        for ann in document['annotations']:
            x, y, w, h = ann['bbox']
            gt[ann['image_id']].append(np.array([x, y, x+w, y+h]))
        predictions = defaultdict(list)
        coco_predictions = []
        previews = []
        for image in document['images']:
            path = image_root / image['file_name']
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f'Cannot read {path}')
            with torch.inference_mode():
                result = inference_detector(model, frame).pred_instances
            for box, score in zip(result.bboxes.cpu().numpy(), result.scores.cpu().numpy()):
                predictions[image['id']].append((box, float(score)))
                x1, y1, x2, y2 = box.tolist()
                coco_predictions.append(dict(image_id=image['id'], category_id=1,
                    bbox=[x1, y1, x2-x1, y2-y1], score=float(score)))
            if len(previews) < 3:
                previews.append(cv2.resize(draw(frame, result), (360, 640)))
        coco = COCO(str(annotation_path))
        if not coco_predictions:
            raise RuntimeError(f'No predictions on {split}')
        evaluator = COCOeval(coco, coco.loadRes(coco_predictions), 'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        tp, fp, fn = match_counts(predictions, gt)
        precision, recall = tp / max(tp+fp, 1), tp / max(tp+fn, 1)
        report.append(f'{split}: precision={precision:.4f}, recall={recall:.4f}, TP={tp}, FP={fp}, FN={fn} '
                      f'(confidence>=0.30, IoU>=0.50); AP50={evaluator.stats[1]:.4f}; mAP50-95={evaluator.stats[0]:.4f}')
        if not cv2.imwrite(str(output / f'{split}_predictions.jpg'), np.hstack(previews)):
            raise RuntimeError('Cannot save evaluation preview')
    if args.video is None:
        report.append('Evaluation completed. Visual review is required before model qualification.')
        (output / 'evaluation.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
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
                previews.append(cv2.resize(draw(frame, instances), (300, 540)))
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
    print('\n'.join(report))


if __name__ == '__main__':
    main()
