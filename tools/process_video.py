"""Process one complete video through every installed TensorRT detector."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    from app.video import process_video
    last_stage = None

    def progress(stage, count, total):
        nonlocal last_stage
        if stage != last_stage or count % 100 == 0 or count == total:
            print(f'{stage}: {count}/{total}', flush=True)
            last_stage = stage

    started = time.perf_counter()
    original, output, detections = process_video(args.source, args.output, args.models, progress)
    if 'torch' in sys.modules:
        raise RuntimeError('Production processing unexpectedly imported PyTorch')
    print(f'PASS: {output.frames} frames; {output.width}x{output.height}; '
          f'{float(output.fps):.6f} FPS; {float(output.duration):.6f} seconds; '
          f'{detections} detections; elapsed {time.perf_counter()-started:.1f}s; no PyTorch.')
    print(f'Preserved all presentation timestamps: {original.timeline_hash == output.timeline_hash}')
    print(f'Output: {args.output.resolve()}')


if __name__ == '__main__':
    main()
