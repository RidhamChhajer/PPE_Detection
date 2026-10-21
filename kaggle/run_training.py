"""Validate and train one RTMDet-L detector using Kaggle /input and /working."""
from pathlib import Path
import os
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {'goggles': ROOT/'configs/goggles_rtmdet_l.py',
           'footwear': ROOT/'configs/footwear_rtmdet_l.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(CONFIGS), required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = Path('/kaggle/working') / f'{args.task}-rtmdet-l-v2'
    command = [sys.executable, str(ROOT/'tools/train.py'), '--task', args.task,
               '--config', str(CONFIGS[args.task]), '--work-dir', str(output),
               '--annotations', str(ROOT/'data'/args.task/'annotations'),
               '--images', str(ROOT/'data'/args.task/'images'),
               '--model-version', 'v2-l', '--dataset-version', 'v1',
               '--epochs', str(args.epochs)]
    if args.check_only:
        command.append('--check-only')
    if args.resume:
        command.append('--resume')
    # Kaggle sets MPLBACKEND to its notebook-inline backend. That value is
    # invalid for the isolated Matplotlib installation used by MMDetection.
    environment = os.environ.copy()
    environment['MPLBACKEND'] = 'Agg'
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == '__main__':
    main()
