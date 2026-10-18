# ATHÉNA - models/test_sequence.py
# Teste l'analyse de séquence vidéo (computer_vision/analyse_sequence.py).
# Les séries d'angles sont FABRIQUÉES pour vérifier la logique (symétrie, cadence, refus de
# conclure) : elles n'ont aucune valeur scientifique. La détection de pose réelle (MediaPipe)
# n'est pas testée ici car elle exige le modèle .task téléchargé (voir README).
#
# Lancement : python -m models.test_sequence  (depuis la racine du projet)

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from computer_vision.analyse_sequence import analyser_series, lire_frames, COLONNES_SERIE  # noqa: E402


def serie_course(duree_s=8.0, fps=10.0, rom_d_facteur=1.0, bruit=1.0, taux_nan=0.0, pics_aberrants=0, graine=0):
    rng = np.random.default_rng(graine)
    t = np.arange(0, duree_s, 1 / fps)
    f = 85 / 60                                   # 85 foulées/min par jambe ≈ 170 pas/min
    genou_g = 135 + 35 * np.sin(2 * np.pi * f * t) + rng.normal(0, bruit, len(t))
    genou_d = 135 + 35 * rom_d_facteur * np.sin(2 * np.pi * f * t + np.pi) + rng.normal(0, bruit, len(t))
    df = pd.DataFrame({"t": t, "genou_g": genou_g, "genou_d": genou_d,
                       "hanche_g": 172 + rng.normal(0, 2, len(t)), "hanche_d": 172 + rng.normal(0, 2, len(t)),
                       "tronc": 6 + rng.normal(0, 1, len(t)), "vis_g": 0.9, "vis_d": 0.9}, columns=COLONNES_SERIE)
    for _ in range(pics_aberrants):
        i = rng.integers(0, len(t)); df.loc[i, "genou_g"] = rng.choice([20, 179])
    if taux_nan > 0:
        masque = rng.random(len(t)) < taux_nan
        df.loc[masque, [c for c in COLONNES_SERIE if c != "t"]] = np.nan
    return df, fps


def main():
    ok = True

    def verif(nom, cond, detail=""):
        nonlocal ok
        print(f"[{'OK' if cond else 'ÉCHEC'}] {nom} {detail}")
        ok &= bool(cond)

    sym, fps = serie_course()
    r_sym = analyser_series(sym, fps)
    verif("séquence symétrique -> statut ok", r_sym["statut"] == "ok")
    verif("indice de symétrie élevé", r_sym["indice_symetrie"] >= 95, f"({r_sym['indice_symetrie']} %)")
    verif("cadence détectée ≈ 170 pas/min", r_sym["cadence_pas_min"] is not None and abs(r_sym["cadence_pas_min"] - 170) <= 20,
          f"({r_sym['cadence_pas_min']})")

    asym, fps = serie_course(rom_d_facteur=0.5)
    r_asym = analyser_series(asym, fps)
    verif("jambe droite à demi-amplitude -> indice de symétrie < 90", r_asym["indice_symetrie"] < 90, f"({r_asym['indice_symetrie']} %)")
    verif("score plus bas que la séquence symétrique", r_asym["score_technique"] < r_sym["score_technique"],
          f"({r_asym['score_technique']} < {r_sym['score_technique']})")
    verif("explication mentionne le déséquilibre", any("déséquilibrée" in e for e in r_asym["explications"]))

    ecarts = []
    for g in range(8):
        b, fps = serie_course(pics_aberrants=6, graine=g)
        ecarts.append(abs(analyser_series(b, fps)["score_technique"] - r_sym["score_technique"]))
    verif("pics aberrants isolés : score stable sur 8 graines (écart max ≤ 8 pts)", max(ecarts) <= 8, f"(écart max {max(ecarts):.1f})")

    peu, fps = serie_course(taux_nan=0.7, graine=2)
    verif("70 % d'images sans détection -> refus de conclure", analyser_series(peu, fps)["statut"] == "detection_insuffisante")
    verif("séquence trop courte -> refus", analyser_series(sym.head(3), fps)["statut"] == "trop_court")

    # Lecture vidéo : vidéo synthétique de 3 s à 30 i/s, sous-échantillonnée à ~10 i/s
    import cv2
    with tempfile.TemporaryDirectory() as d:
        chemin = str(Path(d) / "test.mp4")
        ecrivain = cv2.VideoWriter(chemin, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
        for k in range(90):
            ecrivain.write(np.full((120, 160, 3), k * 2 % 255, dtype=np.uint8))
        ecrivain.release()
        frames, fps_eff = lire_frames(chemin, fps_cible=10)
        verif("lecture vidéo sous-échantillonnée", 25 <= len(frames) <= 35 and 8 <= fps_eff <= 12, f"({len(frames)} images à {fps_eff:.1f} i/s)")
        try:
            lire_frames(str(Path(d) / "inexistante.mp4"))
            verif("vidéo inexistante -> erreur claire", False)
        except ValueError:
            verif("vidéo inexistante -> erreur claire", True)

    print("\nTous les tests passent." if ok else "\nDes tests ont échoué.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
