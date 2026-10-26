@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo PPE Detection System environment not found in this folder.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\stop.py
pause
