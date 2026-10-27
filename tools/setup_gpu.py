"""Double-click setup: local environment, pinned CUDA runtime, and a GPU operation check."""

import argparse
import os
import shutil
import subprocess
import sys
import struct
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def run(*args):
    print("\nRunning:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def verify_gpu(target_3050=False):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check the NVIDIA driver.")
    gpu = torch.cuda.get_device_properties(0)
    if target_3050 and ("RTX 3050" not in gpu.name.upper() or gpu.total_memory < 5.5 * 1024**3):
        raise RuntimeError(
            f"Expected RTX 3050 6 GB on CUDA device 0; found {gpu.name} "
            f"with {gpu.total_memory / 1024**3:.1f} GB. "
            "Use Setup GPU.cmd for a different supported NVIDIA GPU."
        )
    x = torch.ones((32, 32), device="cuda:0")
    y = x @ x
    torch.cuda.synchronize()
    if y[0, 0].item() != 32:
        raise RuntimeError("CUDA calculation verification failed.")
    print(f"Verified NVIDIA GPU: {gpu.name}, VRAM: {gpu.total_memory / 1024**3:.1f} GB")
    print("CUDA runtime:", torch.version.cuda)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtx3050", action="store_true", help="Require an RTX 3050 with 6 GB VRAM")
    args = parser.parse_args()
    if sys.platform != "win32" or struct.calcsize("P") != 8:
        raise RuntimeError("Setup requires 64-bit Windows and 64-bit Python.")
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Install Python 3.10–3.13 (64-bit), then run Setup GPU.cmd again.")
    smi = shutil.which("nvidia-smi") or str(
        Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "nvidia-smi.exe"
    )
    run(smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    if not PYTHON.exists():
        venv.create(ROOT / ".venv", with_pip=True)
    run(PYTHON, "-m", "pip", "install", "--upgrade", "pip")
    run(
        PYTHON,
        "-m",
        "pip",
        "install",
        "torch==2.9.1",
        "torchvision==0.24.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    )
    run(PYTHON, "-m", "pip", "install", "-r", ROOT / "requirements.txt")
    run(
        PYTHON,
        "-c",
        f"from tools.setup_gpu import verify_gpu; verify_gpu(target_3050={args.rtx3050!r})",
    )
    script = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('"
        + str(ROOT / "PPE Detection System.lnk").replace("'", "''")
        + "');$s.TargetPath='"
        + str(ROOT / "Start PPE Detection System.cmd").replace("'", "''")
        + "';$s.WorkingDirectory='"
        + str(ROOT).replace("'", "''")
        + "';$s.Save()"
    )
    run("powershell", "-NoProfile", "-Command", script)
    upper_pt = ROOT / "models" / "upper.pt"
    lower_pt = ROOT / "models" / "lower.pt"
    if not upper_pt.exists() or not lower_pt.exists():
        print("\nDownloading required YOLO models from release...")
        run(PYTHON, "tools/download_models.py")

    print(
        "\nGPU setup complete. Double-click PPE Detection System.lnk or Start PPE Detection System.cmd."
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("\nSetup stopped:", exc)
        print(
            "No CPU fallback was enabled. Check internet, disk space, Python and the NVIDIA driver."
        )
        sys.exit(1)
