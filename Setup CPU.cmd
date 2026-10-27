@echo off
cd /d "%~dp0"
python tools\setup_cpu.py
if errorlevel 1 (
  echo Setup failed. Read the message above.
) else (
  echo Setup complete.
)
pause
