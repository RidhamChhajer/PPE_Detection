"""Double-click setup for CPU-only systems: local environment, CPU PyTorch, and models."""

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


def verify_cpu():
    import torch
    x = torch.ones((32, 32))
    y = x @ x
    if y[0, 0].item() != 32:
        raise RuntimeError("CPU calculation verification failed.")
    print("Verified PyTorch CPU runtime:", torch.__version__)


def main():
    if sys.platform != "win32" or struct.calcsize("P") != 8:
        raise RuntimeError("Setup requires 64-bit Windows and 64-bit Python.")
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Install Python 3.10-3.13 (64-bit), then run setup again.")

    if not PYTHON.exists():
        print("Creating virtual environment (.venv)...")
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
        "https://download.pytorch.org/whl/cpu",
    )
    run(PYTHON, "-m", "pip", "install", "-r", ROOT / "requirements.txt")
    run(
        PYTHON,
        "-c",
        "from tools.setup_cpu import verify_cpu; verify_cpu()",
    )

    # Check and download models if missing
    upper_pt = ROOT / "models" / "upper.pt"
    lower_pt = ROOT / "models" / "lower.pt"
    if not upper_pt.exists() or not lower_pt.exists():
        print("\nDownloading required YOLO models...")
        run(PYTHON, "tools/download_models.py")

    # Create desktop shortcut
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
    print("\nCPU setup complete! Double-click Start PPE Detection System.cmd to run.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("\nSetup stopped:", exc)
        sys.exit(1)
