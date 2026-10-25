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

        # Vidéo Full HD : images réduites à la lecture (mémoire), puis vidéo annotée lisible par un navigateur
        from computer_vision.analyse_sequence import generer_video_annotee, LARGEUR_ANALYSE
        chemin_hd = str(Path(d) / "fullhd.avi")
        ecrivain = cv2.VideoWriter(chemin_hd, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920, 1080))
        for k in range(90):
            img = np.full((1080, 1920, 3), 40, dtype=np.uint8)
            cv2.rectangle(img, (100 + 15 * k, 400), (300 + 15 * k, 700), (200, 120, 30), -1)
            ecrivain.write(img)
        ecrivain.release()
        frames_hd, _ = lire_frames(chemin_hd, fps_cible=10)
        verif("vidéo 1920 px réduite à la lecture", frames_hd[0].shape[1] == LARGEUR_ANALYSE,
              f"({frames_hd[0].shape[1]} px)")
        annotee = generer_video_annotee(chemin_hd, [None] * len(frames_hd))
        relues = 0
        if annotee:
            sortie = Path(d) / ("annotee.webm" if annotee[1] == "video/webm" else "annotee.mp4")
            sortie.write_bytes(annotee[0])
            cap = cv2.VideoCapture(str(sortie))
            while cap.read()[0]:
                relues += 1
            cap.release()
        verif("vidéo annotée produite et relisible", annotee is not None and relues == len(frames_hd),
              f"({annotee[1] if annotee else 'aucune'}, {relues} images)")

        # Horodatages strictement croissants (exigés par le mode VIDÉO de MediaPipe), images sautées sans décodage
        from computer_vision.analyse_sequence import _iterer_images
        temps = [t for _, t, _ in _iterer_images(chemin, 10, 150)]
        verif("horodatages croissants pour le mode vidéo", len(temps) >= 25 and all(b > a for a, b in zip(temps, temps[1:])),
              f"({len(temps)} images, {temps[0]:.0f} -> {temps[-1]:.0f} ms)")

    # Tracé du squelette : épaisseur proportionnelle à l'image, points mal vus jamais dessinés
    from computer_vision.pose_analysis import dessiner_squelette, Point
    pose = [Point(0.5, 0.5, 0.0)] * 33
    for idx, (x, y) in {11: (0.45, 0.3), 23: (0.45, 0.55), 25: (0.47, 0.75), 27: (0.45, 0.95),
                        12: (0.55, 0.3), 24: (0.55, 0.55), 26: (0.53, 0.75), 28: (0.55, 0.95)}.items():
        pose[idx] = Point(x, y, 0.95)
    pose[28] = Point(0.55, 0.95, 0.1)                   # cheville droite cachée
    def pixels_traces(taille):
        fond = np.zeros((taille, taille, 3), dtype=np.uint8)
        return int((dessiner_squelette(fond, pose, "Course à pied").max(axis=2) > 0).sum()), fond
    petit, _ = pixels_traces(400)
    grand, _ = pixels_traces(2000)
    # Image 5 fois plus grande : un trait d'épaisseur FIXE donnerait ~5 fois plus de pixels, un trait
    # proportionnel bien plus (longueur ET épaisseur multipliées).
    verif("trait proportionnel à la taille de l'image", grand > 8 * petit, f"({petit} px -> {grand} px)")
    fond = np.zeros((400, 400, 3), dtype=np.uint8)
    img = dessiner_squelette(fond, pose)
    autour_cheville = img[370:390, 210:230].max()        # cheville droite cachée (0,55 ; 0,95)
    verif("point peu visible non dessiné", autour_cheville == 0)

    # Squat : jambe tendue (personne debout) = pas de jugement, alors que la course juge le même genou
    from computer_vision.pose_analysis import COULEUR_OK, COULEUR_LIMITE, COULEUR_HORS
    def couleurs_jugement(mouvement):
        img = dessiner_squelette(np.zeros((400, 400, 3), dtype=np.uint8), pose, mouvement)
        return sum(int((img == np.array(c, dtype=np.uint8)).all(axis=2).sum())
                   for c in (COULEUR_OK, COULEUR_LIMITE, COULEUR_HORS))
    debout_squat, debout_course = couleurs_jugement("Squat"), couleurs_jugement("Course à pied")
    verif("squat debout : tracé neutre (non jugé)", debout_squat == 0 and debout_course > 0,
          f"({debout_squat} px jugés en squat, {debout_course} px en course)")

    print("\nTous les tests passent." if ok else "\nDes tests ont échoué.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    for _flux in (sys.stdout, sys.stderr):   # console Windows (cp1252) : « ≈ » faisait planter l'affichage
        try:
            _flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()
