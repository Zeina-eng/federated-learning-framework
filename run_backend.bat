@echo off
echo ======================================================================
echo Starting Federated Learning Framework Python Backend (FastAPI + PyTorch)
echo ======================================================================
cd /d "%~dp0backend"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause
