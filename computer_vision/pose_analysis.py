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
# Le modèle PoseLandmarker « full » (~9 Mo, plus précis que « lite ») doit être présent
# localement dans artifacts/. `python -m computer_vision.pose_analysis --setup` le
# télécharge une fois pour toutes ; au lancement du dashboard, aucune connexion
# internet n'est nécessaire si ce fichier existe déjà. Le modèle « lite » sert de secours.

import sys
import urllib.request
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
# Modèle « full » : nettement plus précis sur les membres partiellement cachés, environ 2 fois
# plus lent que « lite » (≈ 60 ms par image sur un portable : 15 s de vidéo en 10 à 20 s).
MODEL_PATH = ARTIFACTS_DIR / "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_LITE_PATH = ARTIFACTS_DIR / "pose_landmarker_lite.task"   # secours si « full » est absent

VISIBILITE_TRACE = 0.5   # un point moins visible n'est ni dessiné ni utilisé pour noter une photo

# Point de pose simplifié (coordonnées relatives 0-1) : léger, copiable et mis en cache par
# Streamlit, avec les mêmes attributs que les points de MediaPipe.
Point = namedtuple("Point", ["x", "y", "visibility"])

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
    # Hanche bornée à 180° : l'angle mesuré entre 3 points ne peut pas dépasser 180°.
    "Course à pied": {"genou": (155, 175), "hanche": (165, 180), "tronc_max": 10},
    "Réception de saut": {"genou": (90, 140), "hanche": (140, 175), "tronc_max": 20},
    "Squat": {"genou": (70, 110), "hanche": (60, 110), "tronc_max": 30},
}


def _telecharger_modele_si_absent():
    if MODEL_PATH.exists():
        return
    # Téléchargement vers un fichier temporaire puis renommage : une coupure
    # réseau ne laisse jamais un modèle tronqué (qui serait vu comme "présent").
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(tmp))
        tmp.replace(MODEL_PATH)
    finally:
        if tmp.exists():
            tmp.unlink()


def telecharger_modele() -> Path:
    """Télécharge le modèle PoseLandmarker si absent (appelable depuis le dashboard)."""
    _telecharger_modele_si_absent()
    _charger_detecteur.cache_clear()
    return MODEL_PATH


def chemin_modele() -> Path:
    """Modèle utilisé : « full » s'il est présent, sinon « lite » (secours hors-ligne)."""
    if MODEL_PATH.exists() or not MODEL_LITE_PATH.exists():
        _telecharger_modele_si_absent()
        return MODEL_PATH
    return MODEL_LITE_PATH


def _options(mode):
    return mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(chemin_modele())),
        running_mode=mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


@lru_cache(maxsize=1)
def _charger_detecteur():
    """Détecteur pour les PHOTOS (chaque image analysée seule)."""
    return mp_vision.PoseLandmarker.create_from_options(_options(mp_vision.RunningMode.IMAGE))


def creer_detecteur_video():
    """Détecteur pour les VIDÉOS : il SUIT la personne d'une image à l'autre et lisse les points
    (moins de tremblements, moins de sauts vers une autre personne). Un nouveau détecteur par
    vidéo : ses horodatages doivent croître. À fermer après usage (`with ... as d:`)."""
    return mp_vision.PoseLandmarker.create_from_options(_options(mp_vision.RunningMode.VIDEO))


def nom_modele() -> str:
    return "full" if chemin_modele() == MODEL_PATH else "lite (secours)"


def modele_disponible_localement() -> bool:
    """Permet au dashboard de savoir, avant d'appeler le détecteur, si une
    connexion sera nécessaire (et donc d'afficher un message clair)."""
    return MODEL_PATH.exists() or MODEL_LITE_PATH.exists()


def points_simples(landmarks) -> list:
    """Convertit les points MediaPipe en Points (x, y, visibilité) légers."""
    return [Point(float(p.x), float(p.y), float(getattr(p, "visibility", 1.0) or 0.0)) for p in landmarks]


def preparer_image(image_bgr: np.ndarray, cote_max: int = 1280) -> np.ndarray:
    """Réduit une photo de téléphone (4000 px) : analyse plus rapide, et un squelette qui reste
    visible une fois l'image affichée (les points sont en coordonnées relatives)."""
    h, w = image_bgr.shape[:2]
    echelle = cote_max / max(h, w)
    if echelle >= 1:
        return image_bgr
    return cv2.resize(image_bgr, (int(w * echelle), int(h * echelle)), interpolation=cv2.INTER_AREA)


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
    image_bgr = preparer_image(image_bgr)
    detecteur = _charger_detecteur()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    resultat = detecteur.detect(mp_image)

    if not resultat.pose_landmarks:
        return {"detecte": False}

    h, w = image_bgr.shape[:2]
    lms = points_simples(resultat.pose_landmarks[0])

    def pt(idx):
        p = lms[idx]
        return (p.x * w, p.y * h, getattr(p, "visibility", 1.0))

    mesures_gauche = _mesures_cote(pt, "G")
    mesures_droite = _mesures_cote(pt, "D")
    cote_principal = "Gauche" if mesures_gauche["visibilite"] >= mesures_droite["visibilite"] else "Droite"
    mesures_principales = mesures_gauche if cote_principal == "Gauche" else mesures_droite

    # Épaule, hanche, genou ou cheville mal vus même du meilleur côté : les angles seraient
    # inventés par le modèle. On montre le squelette mais on refuse de noter.
    if mesures_principales["visibilite"] < VISIBILITE_TRACE:
        return {"detecte": True, "fiable": False, "landmarks_bruts": lms, "mouvement_analyse": mouvement,
                "message": ("Silhouette détectée, mais épaule, hanche, genou ou cheville mal visibles "
                            f"(visibilité {mesures_principales['visibilite']:.0%}) : pas de score. Reprendre la "
                            "photo de profil, corps entier, sans obstacle devant les jambes.")}

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
        "fiable": True,
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


# Couleurs BGR du tracé : neutre (bras, épaules), puis jugement de l'angle par rapport à la
# plage de référence du mouvement (bleu/orange plutôt que vert/rouge : lisible par tous).
COULEUR_NEUTRE = (235, 235, 235)
COULEUR_OK = (196, 142, 27)        # bleu ATHÉNA : angle dans la plage de référence
COULEUR_LIMITE = (34, 160, 240)    # orange : à moins de 15° de la plage
COULEUR_HORS = (61, 40, 192)       # rouge : franchement hors plage
SEGMENTS_JAMBE = {"G": ((23, 25), (25, 27)), "D": ((24, 26), (26, 28))}   # cuisse, tibia
SEGMENT_TRONC = {"G": (11, 23), "D": (12, 24)}
# Squat et réception sont notés au bas du mouvement : jambe encore tendue (genou au-delà de la plage
# + cette marge), l'image n'est pas jugée, sinon une personne debout s'afficherait en rouge.
MOUVEMENTS_EN_FLEXION = ("Squat", "Réception de saut")
MARGE_PHASE_JUGEE = 20
# Deux genoux plus proches que cette fraction de l'image (vue de profil) : une seule série d'étiquettes.
DISTANCE_ETIQUETTES = 0.08


def _couleur_jugement(angle: float, plage: tuple) -> tuple:
    lo, hi = plage
    ecart = max(lo - angle, angle - hi, 0)
    return COULEUR_OK if ecart == 0 else (COULEUR_LIMITE if ecart <= 15 else COULEUR_HORS)


def _texte(img, texte, org, echelle, couleur):
    """Texte lisible sur n'importe quel fond : pavé sombre derrière."""
    epaisseur = max(1, int(round(echelle * 2)))
    (tw, th), base = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, echelle, epaisseur)
    x = int(np.clip(org[0], 0, max(0, img.shape[1] - tw - 4)))
    y = int(np.clip(org[1], th + 4, img.shape[0] - 4))
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base), (25, 25, 25), -1)
    cv2.putText(img, texte, (x, y), cv2.FONT_HERSHEY_SIMPLEX, echelle, couleur, epaisseur, cv2.LINE_AA)


def dessiner_squelette(image_bgr: np.ndarray, landmarks, mouvement: str | None = None) -> np.ndarray:
    """Dessine le squelette détecté sur l'image.

    - épaisseur proportionnelle à la taille de l'image (visible sur une photo de 4000 px comme
      sur une vidéo de 640 px) ;
    - les points peu visibles (membre caché, hors champ) ne sont PAS dessinés : on n'affiche
      pas ce que le modèle devine ;
    - si `mouvement` est donné : angles du genou et de la hanche écrits à côté des articulations,
      cuisse/tibia/tronc colorés selon la plage de référence (bleu = dans la plage, orange =
      proche, rouge = hors plage). Squat et réception : jambe tendue = pas de jugement (tracé
      neutre), comme le score qui n'est calculé qu'au bas du mouvement ;
    - vue de profil (genoux superposés) : étiquettes du côté le plus visible seulement."""
    image_annotee = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    trait = max(2, int(round(min(h, w) / 200)))
    rayon = trait + 2
    echelle_texte = max(0.45, min(h, w) / 900)

    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    visible = [(getattr(lm, "visibility", 1.0) or 0.0) >= VISIBILITE_TRACE for lm in landmarks]
    couleurs = {}
    etiquettes = []

    reference = MOUVEMENTS_REFERENCE.get(mouvement) if mouvement else None
    if reference:
        def pt(idx):
            return (landmarks[idx].x * w, landmarks[idx].y * h, getattr(landmarks[idx], "visibility", 1.0))
        par_cote = {}
        for cote in ("G", "D"):
            m = _mesures_cote(pt, cote)
            if m["visibilite"] < VISIBILITE_TRACE:
                continue
            jambe_tendue = (mouvement in MOUVEMENTS_EN_FLEXION
                            and m["angle_genou"] > reference["genou"][1] + MARGE_PHASE_JUGEE)
            if jambe_tendue:
                c_genou = c_hanche = COULEUR_NEUTRE
            else:
                c_genou = _couleur_jugement(m["angle_genou"], reference["genou"])
                c_hanche = _couleur_jugement(m["angle_hanche"], reference["hanche"])
            cuisse, tibia = SEGMENTS_JAMBE[cote]
            couleurs[cuisse], couleurs[tibia] = c_genou, c_genou
            couleurs[SEGMENT_TRONC[cote]] = c_hanche
            genou, hanche = cuisse[1], cuisse[0]
            par_cote[cote] = (m["visibilite"], [(f"genou {m['angle_genou']:.0f}", points[genou], c_genou),
                                                (f"hanche {m['angle_hanche']:.0f}", points[hanche], c_hanche)])
        if len(par_cote) == 2:
            (xg, yg), (xd, yd) = points[SEGMENTS_JAMBE["G"][0][1]], points[SEGMENTS_JAMBE["D"][0][1]]
            if np.hypot(xg - xd, yg - yd) < DISTANCE_ETIQUETTES * min(h, w):
                plus_visible = max(par_cote, key=lambda c: par_cote[c][0])
                par_cote = {plus_visible: par_cote[plus_visible]}
        for _, etiq in par_cote.values():
            etiquettes.extend(etiq)

    for i, j in CONNEXIONS_SQUELETTE:
        if visible[i] and visible[j]:
            # liseré sombre puis couleur : le trait reste visible sur une pelouse comme sur un mur blanc
            cv2.line(image_annotee, points[i], points[j], (20, 20, 20), trait + 2, cv2.LINE_AA)
            cv2.line(image_annotee, points[i], points[j], couleurs.get((i, j), COULEUR_NEUTRE), trait, cv2.LINE_AA)
    for i, j in CONNEXIONS_SQUELETTE:
        for k in (i, j):
            if visible[k]:
                cv2.circle(image_annotee, points[k], rayon, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(image_annotee, points[k], max(1, rayon - 1), (255, 255, 255), -1, cv2.LINE_AA)
    # Variantes claires des 3 couleurs pour le texte, lisibles sur le pavé sombre.
    texte_clair = {COULEUR_OK: (240, 200, 120), COULEUR_LIMITE: (80, 190, 255), COULEUR_HORS: (120, 120, 255),
                   COULEUR_NEUTRE: COULEUR_NEUTRE}
    for texte, (x, y), couleur in etiquettes:
        _texte(image_annotee, texte, (x + 2 * rayon, y), echelle_texte, texte_clair[couleur])
    return image_annotee


if __name__ == "__main__":
    # python -m computer_vision.pose_analysis --setup
    # À lancer une fois (avec internet) avant une démo hors-ligne.
    if "--setup" in sys.argv:
        print(f"Téléchargement du modèle PoseLandmarker (full, ~9 Mo) vers {MODEL_PATH} ...")
        _telecharger_modele_si_absent()
        print("OK — le module fonctionne maintenant sans connexion internet.")
    else:
        print("Usage : python -m computer_vision.pose_analysis --setup")
