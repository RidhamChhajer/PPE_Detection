"""Inventory this environment and verify pretrained RTMDet-S on CUDA.

Run with the training environment. Outputs stay in artifacts/environment.
This verifies the framework only; the COCO model is not a PPE detector.
"""

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet_s_8xb32-300e_coco/"
    "rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth"
)
DEMO = "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/demo/demo.jpg"


def download(url, destination, hash_prefix=None):
    """Use an atomic download; verify the publisher's checkpoint hash prefix."""
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as source:
                with temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    if hash_prefix:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if not digest.startswith(hash_prefix):
            raise RuntimeError(f"Checkpoint checksum mismatch: {destination}")


def command_version(executable, arguments):
    if not executable:
        return "Not installed or not found in PATH"
    result = subprocess.run(
        [str(executable), *arguments], capture_output=True, text=True,
        timeout=30, check=False,
    )
    return (result.stdout + result.stderr).strip() or f"Exit code {result.returncode}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=Path, help="Local test image")
    inputs.add_argument("--video", type=Path, help="Test the first decoded video frame")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "artifacts" / "environment"
    output.mkdir(parents=True, exist_ok=True)
    lines = []

    def record(message):
        lines.append(str(message))
        print(message, flush=True)

    try:
        record(f"Checked: {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
        record(f"Platform: {platform.platform()}")
        record(f"Python: {platform.python_version()} ({sys.executable})")
        smi = shutil.which("nvidia-smi")
        fallback = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvidia-smi.exe"
        if not smi and fallback.exists():
            smi = fallback
        record("NVIDIA driver / driver-supported CUDA:\n" + command_version(smi, []))
        record("CUDA toolkit compiler:\n" + command_version(shutil.which("nvcc"), ["--version"]))
        for package in ("torch", "torchvision", "mmengine", "mmcv", "mmdet", "tensorrt"):
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "Not installed"
            record(f"{package}: {version}")
        if args.inventory_only:
            record("Inventory only; inference acceptance has NOT been tested.")
            return 0
        if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
            raise RuntimeError("Run this tool with .venv/Scripts/python.exe")

        import cv2
        import mmdet
        import torch
        from mmcv.ops import get_compiling_cuda_version, nms
        from mmdet.apis import inference_detector, init_detector

        record(f"PyTorch CUDA runtime: {torch.version.cuda}")
        record(f"MMCV compiled CUDA: {get_compiling_cuda_version()}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible to PyTorch; CPU fallback is not accepted.")
        record(f"CUDA device: {torch.cuda.get_device_name(0)}")
        record(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device="cuda:0")
        scores = torch.tensor([0.9, 0.8], device="cuda:0")
        _, kept = nms(boxes, scores, 0.5)
        if kept.tolist() != [0]:
            raise RuntimeError("MMCV CUDA NMS returned unexpected results")
        record("MMCV CUDA NMS: PASS")

        if args.video:
            capture = cv2.VideoCapture(str(args.video.resolve()))
            try:
                ok, frame = capture.read()
            finally:
                capture.release()
            if not ok:
                raise ValueError(f"Cannot decode a frame from {args.video}")
            record(f"Input: first frame of {args.video.resolve()}")
        else:
            image_path = args.image
            if image_path is None:
                image_path = output / "demo.jpg"
                download(DEMO, image_path)
            frame = cv2.imread(str(image_path.resolve()))
            if frame is None:
                raise ValueError(f"Cannot decode image {image_path}")
            record(f"Input: {image_path.resolve()}")

        checkpoint = output / CHECKPOINT.rsplit("/", 1)[1]
        download(CHECKPOINT, checkpoint, "387a891e")
        record(f"Checkpoint SHA256: {hashlib.sha256(checkpoint.read_bytes()).hexdigest()}")
        config = Path(mmdet.__file__).parent / ".mim/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py"
        if not config.is_file():
            raise FileNotFoundError(f"Packaged RTMDet-S config missing: {config}")
        model = init_detector(str(config), str(checkpoint), device="cuda:0")
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError("Model was not loaded on CUDA")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = inference_detector(model, frame)
        torch.cuda.synchronize()
        record(f"First inference: {(time.perf_counter() - started) * 1000:.1f} ms (not a throughput benchmark)")
        prediction = result.pred_instances
        if not torch.isfinite(prediction.bboxes).all() or not torch.isfinite(prediction.scores).all():
            raise RuntimeError("Non-finite detector outputs")
        if ((prediction.scores < 0) | (prediction.scores > 1)).any():
            raise RuntimeError("Confidence scores outside [0, 1]")
        if (prediction.bboxes[:, 2:] < prediction.bboxes[:, :2]).any():
            raise RuntimeError("Invalid box geometry")
        selected = prediction[prediction.scores >= 0.3].cpu()
        if not len(selected):
            raise RuntimeError("No detections above 0.3; use a clearly visible COCO object for this smoke test")
        classes = model.dataset_meta["classes"]
        for box, score, label in zip(selected.bboxes, selected.scores, selected.labels):
            x1, y1, x2, y2 = (int(value) for value in box)
            text = f"{classes[int(label)]} {float(score):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (max(0, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        image_output = output / "rtmdet_s_smoke.jpg"
        if not cv2.imwrite(str(image_output), frame):
            raise OSError(f"Cannot write {image_output}")
        record(f"Detections >= 0.30: {len(selected)}")
        record(f"Peak GPU tensor allocation: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
        record(f"Annotated image: {image_output}")
        record("PASS: PyTorch CUDA and pretrained RTMDet-S inference verified. PPE accuracy is not evaluated.")
        return 0
    except Exception as error:
        record(f"FAIL: {type(error).__name__}: {error}")
        return 1
    finally:
        report_name = "inventory.txt" if args.inventory_only else "environment.txt"
        (output / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
