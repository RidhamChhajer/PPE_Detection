@echo off
cd /d "%~dp0"
set "PYTHONPYCACHEPREFIX=%~dp0TEMPORARY\pycache"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup GPU.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\launch.py
