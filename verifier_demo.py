# ATHÉNA - verifier_demo.py
# Check-list "feu vert / feu rouge" à lancer la VEILLE de la démo (avec internet) puis le matin
# (sans internet) : python verifier_demo.py
# Vérifie les fichiers, charge les modèles, lance les suites de tests et signale tout ce qui
# dépendrait du réseau le jour J.

import importlib
import os
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE))

# La console Windows (cp1252) ne sait pas afficher les ✅ / ❌ : sans cela, le
# script plante dès la première ligne quand lancer_demo.bat l'exécute.
for flux in (sys.stdout, sys.stderr):
    try:
        flux.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

resultats = []   # (statut, message) ; statut in {"OK", "AVERT", "ECHEC"}


def noter(statut, msg):
    resultats.append((statut, msg))
    icone = {"OK": "✅", "AVERT": "⚠️ ", "ECHEC": "❌"}[statut]
    print(f"{icone} {msg}")


# 1. Dépendances
for mod, requis in [("streamlit", True), ("plotly", True), ("pandas", True), ("numpy", True), ("sklearn", True),
                    ("scipy", True), ("joblib", True), ("matplotlib", True), ("reportlab", True),
                    ("cv2", True), ("mediapipe", True)]:   # l'analyse vidéo fait partie de la démo
    try:
        importlib.import_module(mod)
        noter("OK", f"Dépendance {mod}")
    except Exception:  # noqa: BLE001
        noter("ECHEC" if requis else "AVERT",
              f"Dépendance {mod} manquante" + ("" if requis else " (analyse vidéo indisponible)"))

# 2. Modèles
for f in ["injury_risk_model.joblib", "fatigue_model.joblib", "metadata.json"]:
    noter("OK" if (RACINE / "models" / "artifacts" / f).exists() else "ECHEC", f"Fichier modèle {f}")
try:
    from models.injury_risk import _charger_modeles
    # Sans réentraînement : un échec doit être VU ici, pas réparé en silence par un nouveau modèle.
    _, _, meta_modele = _charger_modeles(reentrainer_si_echec=False)
    if meta_modele.get("avertissement_version"):
        noter("AVERT", meta_modele["avertissement_version"])
    else:
        noter("OK", "Modèles chargés (compatibles avec cette version de scikit-learn)")
except Exception as e:  # noqa: BLE001
    noter("ECHEC", "Modèles non chargeables (version de scikit-learn différente ?). "
                   f"Solution : python -m models.train_injury_model — détail : {str(e)[:120]}")

# 3. Modèle de pose (analyse vidéo hors-ligne) : présent ET réellement exécutable
pose_full = RACINE / "computer_vision" / "artifacts" / "pose_landmarker_full.task"
pose_lite = RACINE / "computer_vision" / "artifacts" / "pose_landmarker_lite.task"
if pose_full.exists() and pose_full.stat().st_size > 5_000_000:
    noter("OK", "Modèle de pose « full » présent en local (analyse vidéo possible sans internet)")
elif pose_lite.exists() and pose_lite.stat().st_size > 1_000_000:
    noter("AVERT", "Seul le modèle de pose « lite » (moins précis) est présent : lancer UNE FOIS avec internet "
                   "`python -m computer_vision.pose_analysis --setup`")
else:
    noter("ECHEC", "Modèle de pose ABSENT : lancer UNE FOIS avec internet "
                   "`python -m computer_vision.pose_analysis --setup`, sinon l'analyse vidéo échouera hors-ligne")
if pose_full.exists() or pose_lite.exists():
    try:
        import numpy as np
        from computer_vision.pose_analysis import analyser_image, creer_detecteur_video, nom_modele
        analyser_image(np.full((480, 640, 3), 128, dtype=np.uint8))       # image vide : le détecteur doit tourner
        with creer_detecteur_video():
            pass
        noter("OK", f"Détecteur de pose chargé et exécuté (modèle {nom_modele()}, modes photo et vidéo)")
    except Exception as e:  # noqa: BLE001
        noter("ECHEC", f"Le détecteur de pose ne s'exécute pas : {str(e)[:150]}")
    # Démo : si des médias d'exemple sont fournis, une personne doit y être détectée.
    import cv2
    for photo in sorted((RACINE / "demo_media").glob("*.jp*g")) + sorted((RACINE / "demo_media").glob("*.png")):
        img = cv2.imread(str(photo))
        detecte = img is not None and analyser_image(img).get("detecte")
        noter("OK" if detecte else "AVERT", f"Photo de démo {photo.name} : "
              + ("silhouette détectée" if detecte else "AUCUNE silhouette détectée, choisir une autre photo"))

# Polices et logo servis en local (dashboard/static + .streamlit/config.toml)
statique = RACINE / "dashboard" / "static"
polices = list((statique / "fonts").glob("*.woff2"))
if len(polices) >= 7 and (statique / "logo_guindy.png").exists() and (statique / "logo_athena.png").exists() and (RACINE / ".streamlit" / "config.toml").exists():
    noter("OK", f"Polices ({len(polices)} fichiers) et logos présents en local (affichage identique sans internet)")
else:
    noter("AVERT", "Polices ou logo absents de dashboard/static (ou .streamlit/config.toml manquant) : "
                   "l'app s'affichera avec la police par défaut et sans logo")

# 4. Suites de tests
for mod in ["models.test_robustesse", "models.test_donnees_reelles", "models.test_saisie", "models.test_sequence",
            "models.test_demo", "models.test_testeur_csv"]:
    if not (RACINE / (mod.replace(".", "/") + ".py")).exists():
        noter("AVERT", f"Test {mod} introuvable")
        continue
    r = subprocess.run([sys.executable, "-m", mod], cwd=RACINE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    noter("OK" if r.returncode == 0 else "ECHEC", f"Tests {mod}")
    if r.returncode != 0:
        print("   ", (r.stdout + r.stderr).strip().splitlines()[-3:])

echecs = [m for s, m in resultats if s == "ECHEC"]
print("\n" + ("🟢 FEU VERT — démo prête." if not echecs else f"🔴 FEU ROUGE — {len(echecs)} problème(s) à régler."))
sys.exit(1 if echecs else 0)
