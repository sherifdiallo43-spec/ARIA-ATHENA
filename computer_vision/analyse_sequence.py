# ATHÉNA - computer_vision/analyse_sequence.py
# Analyse biomécanique d'une SÉQUENCE (vidéo courte de 5 à 15 s) : au lieu d'une photo
# unique, on suit les angles genou/hanche/tronc image par image, on mesure l'amplitude
# de mouvement, l'asymétrie gauche/droite (moyenne et pic), la cadence, et on produit un
# score technique explicable.
#
# Deux couches volontairement séparées :
#   1. extraire_series_depuis_video() : lecture vidéo + MediaPipe (nécessite le modèle de pose)
#   2. analyser_series()              : calcul PUR sur une table d'angles (aucun modèle requis,
#                                       donc entièrement testable : voir models/test_sequence.py)
#
# Limites assumées : démonstrateur 2D (une seule caméra), angles projetés dans le plan de
# l'image ; vue de profil recommandée pour genou/hanche, vue de face/dos pour l'asymétrie.
# Ce n'est pas un outil médical certifié.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from computer_vision.pose_analysis import (
    MOUVEMENTS_REFERENCE, _score_vs_reference,
)

COLONNES_SERIE = ["t", "genou_g", "genou_d", "hanche_g", "hanche_d", "tronc", "vis_g", "vis_d"]
VISIBILITE_MIN = 0.4
DETECTION_MIN = 0.5     # en dessous de 50 % d'images détectées, on refuse de conclure


# --------------------------------------------------------------- extraction ----

def lire_frames(chemin_video: str | Path, fps_cible: float = 10.0, max_frames: int = 150):
    """Lit une vidéo en sous-échantillonnant à ~fps_cible. Retourne (frames_bgr, fps_effectif)."""
    import cv2
    cap = cv2.VideoCapture(str(chemin_video))
    if not cap.isOpened():
        raise ValueError("Vidéo illisible ou format non pris en charge.")
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pas = max(1, int(round(fps_source / fps_cible)))
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % pas == 0:
            frames.append(frame)
        i += 1
    cap.release()
    if len(frames) < 5:
        raise ValueError("Vidéo trop courte (moins de 5 images exploitables).")
    return frames, fps_source / pas


def extraire_series_depuis_video(chemin_video: str | Path, fps_cible: float = 10.0,
                                 max_frames: int = 150, rappel_progression=None) -> tuple[pd.DataFrame, float, list]:
    """Détecte la pose sur chaque image. Retourne (table d'angles, fps effectif, landmarks par image)."""
    import cv2
    import mediapipe as mp
    from computer_vision.pose_analysis import (
        _charger_detecteur, _mesures_cote, LM_EPAULE_G, LM_EPAULE_D, LM_HANCHE_G, LM_HANCHE_D,
    )
    frames, fps = lire_frames(chemin_video, fps_cible, max_frames)
    detecteur = _charger_detecteur()
    lignes, landmarks_par_image = [], []
    for k, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detecteur.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if rappel_progression:
            rappel_progression((k + 1) / len(frames))
        if not res.pose_landmarks:
            lignes.append({"t": k / fps, **{c: np.nan for c in COLONNES_SERIE[1:]}})
            landmarks_par_image.append(None)
            continue
        h, w = frame.shape[:2]
        lms = res.pose_landmarks[0]
        pt = lambda idx: (lms[idx].x * w, lms[idx].y * h, getattr(lms[idx], "visibility", 1.0))  # noqa: E731
        g, d = _mesures_cote(pt, "G"), _mesures_cote(pt, "D")
        # côté de référence pour le tronc = le plus visible
        ep, ha = (pt(LM_EPAULE_G), pt(LM_HANCHE_G)) if g["visibilite"] >= d["visibilite"] else (pt(LM_EPAULE_D), pt(LM_HANCHE_D))
        tronc = float(np.degrees(np.arctan2(abs(ep[0] - ha[0]), abs(ep[1] - ha[1]) + 1e-9)))
        lignes.append({"t": k / fps, "genou_g": g["angle_genou"], "genou_d": d["angle_genou"],
                       "hanche_g": g["angle_hanche"], "hanche_d": d["angle_hanche"], "tronc": tronc,
                       "vis_g": g["visibilite"], "vis_d": d["visibilite"]})
        landmarks_par_image.append(lms)
    return pd.DataFrame(lignes, columns=COLONNES_SERIE), float(fps), landmarks_par_image


# ------------------------------------------------------------------ analyse ----

SEUIL_ABERRANT_DEG = 25.0   # écart à la médiane locale au-delà duquel un angle est jugé aberrant


def _lisser(x: pd.Series, fenetre: int = 5) -> pd.Series:
    """Filtre de Hampel : ne remplace QUE les valeurs aberrantes (écart > 25° à la médiane locale).

    Une médiane glissante appliquée partout rabote les pics d'une foulée (testé : -30 % d'amplitude
    à 10 images/s) ; ici les points normaux restent intacts et l'amplitude est préservée."""
    med = x.rolling(fenetre, center=True, min_periods=3).median()
    aberrant = (x - med).abs() > SEUIL_ABERRANT_DEG
    return x.where(~aberrant, med)


def _pics_flexion(angle: pd.Series, fps: float, prominence: float = 12.0) -> np.ndarray:
    """Indices des flexions maximales du genou (minima de l'angle), distance min ~0.25 s."""
    a = angle.interpolate(limit_direction="both").values
    pics, _ = find_peaks(-a, prominence=prominence, distance=max(1, int(0.25 * fps)))
    return pics


def analyser_series(serie: pd.DataFrame, fps: float, mouvement: str = "Course à pied") -> dict:
    """Analyse pure d'une série d'angles. Ne lève pas d'exception sur des données dégradées."""
    ref = MOUVEMENTS_REFERENCE.get(mouvement, MOUVEMENTS_REFERENCE["Course à pied"])
    n = len(serie)
    if n < 5:
        return {"statut": "trop_court", "message": "Moins de 5 images : séquence inexploitable."}

    detectee = serie[["genou_g", "genou_d", "hanche_g", "hanche_d"]].notna().any(axis=1)
    taux_detection = float(detectee.mean())
    deux_cotes = (serie["vis_g"] > VISIBILITE_MIN) & (serie["vis_d"] > VISIBILITE_MIN)
    taux_deux_cotes = float(deux_cotes.mean())

    if taux_detection < DETECTION_MIN:
        return {"statut": "detection_insuffisante", "taux_detection": round(taux_detection, 2),
                "message": (f"Silhouette détectée sur seulement {taux_detection:.0%} des images "
                            f"(minimum {DETECTION_MIN:.0%}). Filmer en plan large, corps entier, bonne lumière.")}

    s = serie.copy()
    for c in ["genou_g", "genou_d", "hanche_g", "hanche_d", "tronc"]:
        s[c] = _lisser(s[c])
    # on n'utilise un côté que sur les images où il est assez visible
    for cote in ("g", "d"):
        masque = s[f"vis_{cote}"] <= VISIBILITE_MIN
        s.loc[masque, [f"genou_{cote}", f"hanche_{cote}"]] = np.nan

    # --- amplitude de mouvement du genou par côté
    def _rom(col):
        v = s[col].dropna()   # percentiles 5-95 : robuste aux aberrations résiduelles
        return float(np.percentile(v, 95) - np.percentile(v, 5)) if len(v) >= 3 else np.nan

    def _flexion_max(col):
        v = s[col].dropna()
        return float(np.percentile(v, 5)) if len(v) >= 3 else np.nan

    rom_g, rom_d = _rom("genou_g"), _rom("genou_d")

    # --- asymétrie. Deux mesures selon le mouvement :
    #   * mouvements bilatéraux en phase (squat, réception de saut) : écart INSTANTANÉ gauche/droite ;
    #   * course : les jambes sont en opposition de phase, l'écart instantané n'a donc aucun sens
    #     (il vaut ~l'amplitude même pour une foulée parfaite). On compare alors l'amplitude (ROM)
    #     et la flexion maximale de chaque jambe.
    bilateral_en_phase = mouvement in ("Squat", "Réception de saut")
    asym_moy = asym_pic = np.nan
    if bilateral_en_phase:
        ecart = (s["genou_g"] - s["genou_d"]).abs()[deux_cotes].dropna()
        if len(ecart) >= 3:
            asym_moy, asym_pic = float(ecart.mean()), float(np.percentile(ecart, 95))
    fm_g, fm_d = _flexion_max("genou_g"), _flexion_max("genou_d")
    ecart_flexion = abs(fm_g - fm_d) if np.isfinite(fm_g) and np.isfinite(fm_d) else np.nan
    if np.isfinite(rom_g) and np.isfinite(rom_d) and max(rom_g, rom_d) > 0:
        indice_symetrie = 100 * min(rom_g, rom_d) / max(rom_g, rom_d)
    else:
        indice_symetrie = np.nan

    # --- cadence (foulée / appuis) : sur le côté le mieux détecté
    meilleur = "genou_g" if s["genou_g"].notna().sum() >= s["genou_d"].notna().sum() else "genou_d"
    pics = _pics_flexion(s[meilleur], fps)
    duree = float(s["t"].iloc[-1] - s["t"].iloc[0]) if n > 1 else 0.0
    cadence = None
    if len(pics) >= 4 and duree > 0:
        # un côté seul = une foulée sur deux : x2 pour obtenir des pas/min (course)
        cadence = round(len(pics) / duree * 60 * 2, 0) if mouvement == "Course à pied" else None

    # --- score technique, calculé aux IMAGES CLÉS (et non sur toute la séquence) :
    # les angles de référence décrivent un moment précis du geste — extension maximale du genou
    # (appui au sol) en course ; flexion maximale (bas du mouvement) en squat / réception de saut.
    # Évaluer chaque image contre cette référence pénaliserait à tort la phase de suspension
    # d'une foulée parfaite (le genou y est naturellement fléchi).
    extension_cle = (mouvement == "Course à pied")

    def _images_cles(col):
        a = s[col].interpolate(limit_direction="both").values
        if len(a) < 5 or np.all(np.isnan(a)):
            return np.array([], dtype=int)
        sens = a if extension_cle else -a
        idx, _ = find_peaks(sens, prominence=12.0, distance=max(1, int(0.25 * fps)))
        return idx

    scores_par_cote, n_evenements = {}, {}
    for cote in ("g", "d"):
        col_k, col_h = f"genou_{cote}", f"hanche_{cote}"
        idx = _images_cles(col_k)
        vals = []
        for i in idx:
            gk, hh, tr = s[col_k].iloc[i], s[col_h].iloc[i], s["tronc"].iloc[i]
            if np.isfinite(gk) and np.isfinite(hh) and np.isfinite(tr):
                vals.append(_score_vs_reference(gk, hh, tr, ref))
        if vals:
            scores_par_cote[cote], n_evenements[cote] = float(np.mean(vals)), len(vals)
    if not scores_par_cote:
        return {"statut": "detection_insuffisante", "taux_detection": round(taux_detection, 2),
                "message": ("Aucun cycle de mouvement exploitable détecté (pas assez de foulées/répétitions "
                            "visibles). Filmer au moins 4 foulées ou 3 répétitions, corps entier, bonne lumière.")}
    score = float(np.mean(list(scores_par_cote.values())))
    moment = "l'extension maximale du genou (appui au sol)" if extension_cle else "la flexion maximale (bas du mouvement)"
    unite = "foulées" if extension_cle else "répétitions"

    penalites = []
    if np.isfinite(indice_symetrie) and indice_symetrie < 90:
        p = min(15.0, (90 - indice_symetrie) * 0.6)
        score -= p
        penalites.append(f"amplitude gauche/droite déséquilibrée (indice de symétrie {indice_symetrie:.0f} %) : -{p:.0f} pts")
    if bilateral_en_phase and np.isfinite(asym_pic) and asym_pic > 15:
        score -= 5
        penalites.append(f"pic d'asymétrie de {asym_pic:.0f}° sur le genou : -5 pts")
    if not bilateral_en_phase and np.isfinite(ecart_flexion) and ecart_flexion > 15:
        score -= 5
        penalites.append(f"flexion maximale du genou différente d'une jambe à l'autre ({ecart_flexion:.0f}°) : -5 pts")
    score = float(np.clip(score, 0, 100))

    niveau = ("Bonne technique" if score >= 80 else
              "Corrections mineures recommandées" if score >= 60 else "Déséquilibre notable détecté")

    # --- explication en français (pas de boîte noire)
    explications = [
        f"Score évalué à {moment} sur {sum(n_evenements.values())} {unite} "
        f"(côté{'s' if len(scores_par_cote) > 1 else ''} "
        f"{' et '.join({'g': 'gauche', 'd': 'droit'}[c] for c in scores_par_cote)}) : "
        f"{np.mean(list(scores_par_cote.values())):.0f}/100 avant pénalités de symétrie.",
    ]
    explications += penalites
    tronc_med = float(s["tronc"].median())
    if tronc_med > ref["tronc_max"]:
        explications.append(f"Tronc incliné de {tronc_med:.0f}° en médiane (repère pour ce mouvement : ≤ {ref['tronc_max']}°).")
    if taux_deux_cotes < 0.5:
        explications.append("Moins de la moitié des images montrent les deux côtés : l'asymétrie est peu fiable. "
                            "Pour l'évaluer, filmer de face ou de dos.")

    return {
        "statut": "ok", "mouvement": mouvement, "n_images": int(n), "duree_s": round(duree, 1),
        "taux_detection": round(taux_detection, 2), "taux_deux_cotes": round(taux_deux_cotes, 2),
        "rom_genou_g": None if not np.isfinite(rom_g) else round(rom_g, 1),
        "rom_genou_d": None if not np.isfinite(rom_d) else round(rom_d, 1),
        "indice_symetrie": None if not np.isfinite(indice_symetrie) else round(indice_symetrie, 1),
        "asymetrie_moyenne_deg": None if not np.isfinite(asym_moy) else round(asym_moy, 1),
        "asymetrie_pic_deg": None if not np.isfinite(asym_pic) else round(asym_pic, 1),
        "ecart_flexion_max_deg": None if not np.isfinite(ecart_flexion) else round(ecart_flexion, 1),
        "nb_flexions_detectees": int(len(pics)), "cadence_pas_min": cadence,
        "score_technique": round(score, 1), "niveau": niveau, "explications": explications,
        "serie_lissee": s,
    }


def analyser_video(chemin_video: str | Path, mouvement: str = "Course à pied",
                   fps_cible: float = 10.0, rappel_progression=None) -> dict:
    """Pipeline complet : vidéo -> pose image par image -> analyse temporelle."""
    serie, fps, landmarks = extraire_series_depuis_video(chemin_video, fps_cible, rappel_progression=rappel_progression)
    resultat = analyser_series(serie, fps, mouvement)
    resultat["landmarks_par_image"] = landmarks
    resultat["fps"] = round(fps, 1)
    return resultat
