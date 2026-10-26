@echo off
cd /d "%~dp0"
python tools\setup_gpu.py --rtx3050
if errorlevel 1 (
  echo Setup failed. Read the message above.
) else (
  echo RTX 3050 6GB setup complete. Run Start PPE Detection System.cmd.
)
pause
