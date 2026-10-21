"""Save review frames and raw-vs-paired footwear predictions for one video.

This is a qualitative audit. It does not create ground-truth labels or compute
accuracy metrics. Use independent, manually annotated footage for those.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.registry import model_specs
from app.runtime import Detector
from app.video import draw_detections, pair_footwear


def detection_record(box):
    return dict(box=[round(box.x1, 1), round(box.y1, 1), round(box.x2, 1),
                     round(box.y2, 1)], class_name=box.class_label,
                confidence=round(box.confidence, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seconds', type=float, nargs='+', required=True)
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    parser.add_argument('--registry', type=Path, default=ROOT/'models/registry.local.json')
    parser.add_argument('--backend', choices=('tensorrt', 'onnx'), default='tensorrt')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory; existing audits are not overwritten')
    if any(second < 0 for second in args.seconds):
        parser.error('Sample times must be nonnegative')

    specs = [spec for spec in model_specs(args.models, args.registry, args.backend)
             if any('SHOE' in label.upper() for label in spec.labels)]
    if len(specs) != 1:
        raise ValueError('Expected exactly one footwear detector')
    spec = specs[0]
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f'Cannot open {args.video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= 0:
        capture.release()
        raise ValueError('Video needs valid FPS and frame count')
    args.output.mkdir(parents=True)
    records, tiles = [], []
    try:
        with Detector(spec) as detector:
            for second in args.seconds:
                frame_index = round(second * fps)
                if frame_index >= count:
                    raise ValueError(f'{second}s is beyond the video duration')
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'Could not read frame {frame_index}')
                raw = detector.detect(frame)
                paired = pair_footwear(raw, spec.labels)
                left, right = frame.copy(), frame.copy()
                draw_detections(left, spec.labels, raw, [(0, 220, 0)] * len(spec.labels))
                draw_detections(right, spec.labels, paired, [(0, 220, 0)] * len(spec.labels))
                tile = np.hstack((left, right))
                cv2.putText(tile, f'{second:g}s  raw (left) / paired (right)',
                            (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
                path = args.output / f'frame_{frame_index:06d}.jpg'
                if not cv2.imwrite(str(path), tile):
                    raise RuntimeError(f'Could not save {path}')
                tiles.append(cv2.resize(tile, (480, 432)))
                records.append(dict(second=second, frame=frame_index,
                                    raw=[detection_record(box) for box in raw],
                                    paired=[detection_record(box) for box in paired],
                                    image=path.name))
    finally:
        capture.release()
    sheet = np.vstack(tiles)
    if not cv2.imwrite(str(args.output/'contact_sheet.jpg'), sheet):
        raise RuntimeError('Could not save contact sheet')
    (args.output/'predictions.json').write_text(json.dumps(dict(
        source=str(args.video.resolve()), model=spec.model_id, fps=fps,
        note='Predictions only; human review and ground truth are still required.',
        frames=records), indent=2), encoding='utf-8')
    print(f'Saved {len(records)} sampled frames to {args.output.resolve()}')


if __name__ == '__main__':
    main()
