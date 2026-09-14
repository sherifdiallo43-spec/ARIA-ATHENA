# computer_vision/pose_analysis.py
#
# Analyse posturale par vision par ordinateur (MediaPipe PoseLandmarker).
# Calcule, pour une image donnée : angle du genou, angle de la hanche,
# inclinaison du tronc, et une comparaison gauche/droite (asymétrie).
#
# Le calibrage de ce qu'est un "bon" angle dépend du mouvement analysé
# (une réception de saut n'a pas la même posture idéale qu'une foulée de
# course ou un squat) — voir MOUVEMENTS_REFERENCE plus bas.
#
# Le modèle PoseLandmarker (~5 Mo) doit être présent localement dans
# artifacts/. `python -m computer_vision.pose_analysis --setup` le
# télécharge une fois pour toutes ; au lancement du dashboard, aucune
# connexion internet n'est nécessaire si ce fichier existe déjà.

import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS_DIR / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Indices des landmarks BlazePose (33 points)
LM_EPAULE_G, LM_EPAULE_D = 11, 12
LM_HANCHE_G, LM_HANCHE_D = 23, 24
LM_GENOU_G, LM_GENOU_D = 25, 26
LM_CHEVILLE_G, LM_CHEVILLE_D = 27, 28

CONNEXIONS_SQUELETTE = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

# Angles de référence par type de mouvement (genou/hanche visés en degrés).
# Valeurs indicatives pour un outil d'aide à la détection de déséquilibres
# grossiers — pas un référentiel biomécanique certifié. Point de départ :
# littérature course à pied (attaque du sol, genou quasi tendu ~160-170°)
# et réception de saut (genou fléchi pour absorber l'impact, cf. prévention
# des blessures du LCA).
MOUVEMENTS_REFERENCE = {
    "Course à pied": {"genou": (155, 175), "hanche": (165, 185), "tronc_max": 10},
    "Réception de saut": {"genou": (90, 140), "hanche": (140, 175), "tronc_max": 20},
    "Squat": {"genou": (70, 110), "hanche": (60, 110), "tronc_max": 30},
}


def _telecharger_modele_si_absent():
    if MODEL_PATH.exists():
        return
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))


@lru_cache(maxsize=1)
def _charger_detecteur():
    _telecharger_modele_si_absent()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def modele_disponible_localement() -> bool:
    """Permet au dashboard de savoir, avant d'appeler le détecteur, si une
    connexion sera nécessaire (et donc d'afficher un message clair)."""
    return MODEL_PATH.exists()


def _angle_3points(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _mesures_cote(pt, cote: str) -> dict:
    """Calcule les angles genou/hanche pour un côté donné ('G' ou 'D')."""
    idx_epaule = LM_EPAULE_G if cote == "G" else LM_EPAULE_D
    idx_hanche = LM_HANCHE_G if cote == "G" else LM_HANCHE_D
    idx_genou = LM_GENOU_G if cote == "G" else LM_GENOU_D
    idx_cheville = LM_CHEVILLE_G if cote == "G" else LM_CHEVILLE_D

    epaule, hanche, genou, cheville = pt(idx_epaule), pt(idx_hanche), pt(idx_genou), pt(idx_cheville)
    visibilite = min(epaule[2], hanche[2], genou[2], cheville[2])

    return {
        "angle_genou": _angle_3points(hanche[:2], genou[:2], cheville[:2]),
        "angle_hanche": _angle_3points(epaule[:2], hanche[:2], genou[:2]),
        "visibilite": float(visibilite),
    }


def _score_vs_reference(angle_genou, angle_hanche, angle_tronc, reference: dict) -> float:
    g_min, g_max = reference["genou"]
    h_min, h_max = reference["hanche"]
    score = 100.0
    if angle_genou < g_min:
        score -= (g_min - angle_genou) * 1.1
    elif angle_genou > g_max:
        score -= (angle_genou - g_max) * 1.1
    if angle_hanche < h_min:
        score -= (h_min - angle_hanche) * 1.0
    elif angle_hanche > h_max:
        score -= (angle_hanche - h_max) * 1.0
    score -= max(0, angle_tronc - reference["tronc_max"]) * 1.4
    return float(np.clip(score, 0, 100))


def analyser_image(image_bgr: np.ndarray, mouvement: str = "Course à pied") -> dict:
    """
    Analyse une image et retourne les métriques biomécaniques du côté le
    mieux visible, la comparaison avec le côté opposé (asymétrie), et un
    score technique calibré sur le type de mouvement choisi.
    """
    detecteur = _charger_detecteur()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    resultat = detecteur.detect(mp_image)

    if not resultat.pose_landmarks:
        return {"detecte": False}

    h, w = image_bgr.shape[:2]
    lms = resultat.pose_landmarks[0]

    def pt(idx):
        p = lms[idx]
        return (p.x * w, p.y * h, getattr(p, "visibility", 1.0))

    mesures_gauche = _mesures_cote(pt, "G")
    mesures_droite = _mesures_cote(pt, "D")
    cote_principal = "Gauche" if mesures_gauche["visibilite"] >= mesures_droite["visibilite"] else "Droite"
    mesures_principales = mesures_gauche if cote_principal == "Gauche" else mesures_droite

    epaule_ref = pt(LM_EPAULE_G if cote_principal == "Gauche" else LM_EPAULE_D)
    hanche_ref = pt(LM_HANCHE_G if cote_principal == "Gauche" else LM_HANCHE_D)
    dx = epaule_ref[0] - hanche_ref[0]
    dy = epaule_ref[1] - hanche_ref[1]
    angle_tronc_vertical = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9)))

    reference = MOUVEMENTS_REFERENCE.get(mouvement, MOUVEMENTS_REFERENCE["Course à pied"])
    score = _score_vs_reference(
        mesures_principales["angle_genou"], mesures_principales["angle_hanche"],
        angle_tronc_vertical, reference,
    )

    # Asymétrie : écart d'angle genou/hanche entre les deux côtés. Fiable
    # seulement si les deux côtés sont raisonnablement visibles (sinon on
    # compare du signal à du bruit).
    deux_cotes_visibles = mesures_gauche["visibilite"] > 0.4 and mesures_droite["visibilite"] > 0.4
    ecart_genou = abs(mesures_gauche["angle_genou"] - mesures_droite["angle_genou"])
    ecart_hanche = abs(mesures_gauche["angle_hanche"] - mesures_droite["angle_hanche"])

    if not deux_cotes_visibles:
        asymetrie = {"mesurable": False}
    elif ecart_genou > 15 or ecart_hanche > 15:
        asymetrie = {"mesurable": True, "niveau": "Notable", "ecart_genou": round(ecart_genou, 1), "ecart_hanche": round(ecart_hanche, 1)}
    elif ecart_genou > 8 or ecart_hanche > 8:
        asymetrie = {"mesurable": True, "niveau": "Légère", "ecart_genou": round(ecart_genou, 1), "ecart_hanche": round(ecart_hanche, 1)}
    else:
        asymetrie = {"mesurable": True, "niveau": "Aucune notable", "ecart_genou": round(ecart_genou, 1), "ecart_hanche": round(ecart_hanche, 1)}

    if asymetrie.get("niveau") == "Notable":
        score = max(0, score - 10)

    if score >= 80:
        niveau = "Bonne technique"
    elif score >= 60:
        niveau = "Corrections mineures recommandées"
    else:
        niveau = "Déséquilibre notable détecté"

    return {
        "detecte": True,
        "mouvement_analyse": mouvement,
        "cote_analyse": cote_principal,
        "angle_genou": round(mesures_principales["angle_genou"], 1),
        "angle_hanche": round(mesures_principales["angle_hanche"], 1),
        "angle_tronc_vertical": round(angle_tronc_vertical, 1),
        "asymetrie": asymetrie,
        "score_technique": round(score, 1),
        "niveau": niveau,
        "landmarks_bruts": lms,
    }


def dessiner_squelette(image_bgr: np.ndarray, landmarks) -> np.ndarray:
    """Dessine le squelette détecté (points + connexions) sur l'image."""
    image_annotee = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for i, j in CONNEXIONS_SQUELETTE:
        cv2.line(image_annotee, points[i], points[j], (212, 245, 0), 2)
    for x, y in points:
        cv2.circle(image_annotee, (x, y), 4, (0, 245, 212), -1)
    return image_annotee


if __name__ == "__main__":
    # python -m computer_vision.pose_analysis --setup
    # À lancer une fois (avec internet) avant une démo hors-ligne.
    if "--setup" in sys.argv:
        print(f"Téléchargement du modèle PoseLandmarker vers {MODEL_PATH} ...")
        _telecharger_modele_si_absent()
        print("OK — le module fonctionne maintenant sans connexion internet.")
    else:
        print("Usage : python -m computer_vision.pose_analysis --setup")
