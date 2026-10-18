@echo off
REM ATHENA - lancement de la demo (Windows).
REM Premiere fois (AVEC internet) : installe l'environnement et telecharge le modele de pose.
cd /d "%~dp0"
if not exist ".venv" (
  echo Creation de l'environnement Python...
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
)
if not exist "computer_vision\artifacts\pose_landmarker_lite.task" (
  echo Telechargement du modele de pose (une seule fois, ~5 Mo)...
  .venv\Scripts\python -m computer_vision.pose_analysis --setup
)
.venv\Scripts\python verifier_demo.py
.venv\Scripts\streamlit run dashboard\app.py
