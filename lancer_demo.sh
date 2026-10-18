#!/usr/bin/env bash
# ATHÉNA - lancement de la démo (Linux / macOS).
# Première fois (AVEC internet) : installe l'environnement et télécharge le modèle de pose.
# Ensuite : fonctionne sans internet.
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "Création de l'environnement Python…"
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
if [ ! -f "computer_vision/artifacts/pose_landmarker_lite.task" ]; then
  echo "Téléchargement du modèle de pose (une seule fois, ~5 Mo)…"
  .venv/bin/python -m computer_vision.pose_analysis --setup || echo "⚠️ Échec (pas d'internet ?) : l'analyse vidéo sera indisponible."
fi
.venv/bin/python verifier_demo.py || echo "⚠️ Des vérifications ont échoué : lire les lignes ❌ ci-dessus."
.venv/bin/streamlit run dashboard/app.py
