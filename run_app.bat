@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo .venv bulunamadi. Once bir kez kurulum yap:
  echo py -3 -m venv .venv
  echo .\.venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)

".venv\Scripts\pythonw.exe" main.py
