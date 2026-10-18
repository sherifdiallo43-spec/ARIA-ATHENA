# ATHÉNA - models/experiment_interaction.py
# Expérience E3 : que devient la profondeur optimale quand le risque contient une INTERACTION (ACWR x sommeil) ?
# Hypothèses et règles de décision : docs/E3_preenregistrement.md (écrit AVANT la première exécution).
# Complète E2 (risque additif -> profondeur 1 optimale) et sert à choisir une profondeur ROBUSTE.
#
# Lancement : python -m models.experiment_interaction      (depuis la racine du projet ; quelques minutes)
# Sortie    : models/artifacts/experiment_interaction.json

import json
import time

import numpy as np
from joblib import Parallel, delayed

from .train_injury_model import _generer_dataset, ARTIFACTS_DIR
from .experiment_profondeur import (risque_latent, evaluer_replicat, _agreger, _diff_apparie, PROFONDEURS)

SORTIE = ARTIFACTS_DIR / "experiment_interaction.json"
FORCES = [0.0, 0.25, 0.5, 1.0]
GRAINES = list(range(2000, 2010))
N = 6000
PREVALENCE_CIBLE = 0.20


def generer_jeu_interaction(n: int, graine: int, c: float):
    """X identique à _generer_dataset ; étiquettes retirées avec un risque additif + terme croisé c*zA*zS."""
    X, _, _ = _generer_dataset(n=n, graine=graine)
    zA = (X[:, 0] - 1.05) / 0.35
    zS = (7.0 - X[:, 3]) / 1.1
    r = risque_latent(X) + c * zA * zS
    rng = np.random.default_rng(10_000 + graine)
    eps = rng.normal(0, 0.9, n)
    lo, hi = 0.0, 10.0                                      # ordonnée b telle que la prévalence attendue vaut 20 %
    for _ in range(60):
        b = (lo + hi) / 2
        if (1 / (1 + np.exp(-(r + eps - b)))).mean() > PREVALENCE_CIBLE:
            lo = b
        else:
            hi = b
    y = rng.binomial(1, 1 / (1 + np.exp(-(r + eps - b))))
    return X, y, r, float(b)


def _replicat(c: float, graine: int) -> dict:
    X, y, r, b = generer_jeu_interaction(N, graine, c)
    out = evaluer_replicat(X, y, r, graine_partition=0, graine_modele=0)
    out["b"] = b
    out["sd_terme_croise"] = float(np.std(c * ((X[:, 0] - 1.05) / 0.35) * ((7.0 - X[:, 3]) / 1.1)))
    return out


def verdicts(res: dict) -> dict:
    cles = [str(c) for c in FORCES]
    d = lambda c, cle, p: res[c]["diff"][cle][p]
    # H5
    h5_ok = all(d(str(c), "gb_fixe", "2_vs_1")["ic95"][0] > 0.005 for c in (0.5, 1.0))
    h5_non = d("1.0", "gb_fixe", "2_vs_1")["ic95"][1] < 0.005
    h5 = "confirmée" if h5_ok else ("contredite" if h5_non else "non concluante")
    # H6
    bornes = {c: d(c, "gb_fixe", "3_vs_2")["ic95"][1] for c in cles}
    contredite = any(d(c, "gb_fixe", "3_vs_2")["ic95"][0] > 0 and d(c, "gb_fixe", "3_vs_2")["moyenne"] >= 0.010 for c in cles)
    h6 = "confirmée" if all(b < 0.010 for b in bornes.values()) else ("contredite" if contredite else "non concluante")
    # H7
    regret = {}
    for c in cles:
        moy = {p: res[c]["agrege"]["gb_fixe"][p]["moyenne"] for p in map(str, PROFONDEURS)}
        best = max(moy.values())
        regret[c] = {p: round(best - v, 4) for p, v in moy.items()}
    r2 = max(regret[c]["2"] for c in cles)
    r1 = max(regret[c]["1"] for c in cles)
    h7 = "confirmée" if (r2 < 0.012 and r2 < r1) else ("contredite" if r2 >= r1 else "non concluante")
    # H8
    h8v = d("0.0", "gb_fixe", "2_vs_1")["ic95"][1]
    prev_ok = all(0.15 <= res[c]["agrege"]["taux_positifs_test"] <= 0.25 for c in cles)
    return {"H5": {"verdict": h5, "ic95_2_vs_1": {c: d(c, "gb_fixe", "2_vs_1") for c in cles}, "seuil": 0.005},
            "H6": {"verdict": h6, "bornes_hautes_3_vs_2": bornes, "seuil": 0.010},
            "H7": {"verdict": h7, "regret_max_d2": r2, "regret_max_d1": r1, "regrets": regret, "seuil": 0.012},
            "H8": {"verdict": "confirmée" if h8v < 0.010 else "contredite", "borne_haute_2_vs_1_c0": h8v},
            "controle_prevalence": "ok" if prev_ok else "ECHEC"}


def main():
    t0 = time.time()
    taches = [(c, g) for c in FORCES for g in GRAINES]
    reps = Parallel(n_jobs=-1)(delayed(_replicat)(c, g) for c, g in taches)
    print(f"Réplicats terminés ({time.time() - t0:.0f} s)", flush=True)
    res = {}
    for c in FORCES:
        lot = [r for (cc, _), r in zip(taches, reps) if cc == c]
        agr = _agreger(lot)
        diff = {}
        for cle in ("gb_fixe", "gb_arret"):
            diff[cle] = {"2_vs_1": _diff_apparie([r[cle]["2"] for r in lot], [r[cle]["1"] for r in lot]),
                         "3_vs_2": _diff_apparie([r[cle]["3"] for r in lot], [r[cle]["2"] for r in lot]),
                         "4_vs_2": _diff_apparie([r[cle]["4"] for r in lot], [r[cle]["2"] for r in lot]),
                         "2_vs_lr": _diff_apparie([r[cle]["2"] for r in lot], [r["lr"] for r in lot]),
                         "1_vs_lr": _diff_apparie([r[cle]["1"] for r in lot], [r["lr"] for r in lot])}
        res[str(c)] = {"agrege": agr, "diff": diff, "sd_terme_croise": round(float(np.mean([r["sd_terme_croise"] for r in lot])), 3),
                       "ordonnee_b": round(float(np.mean([r["b"] for r in lot])), 3)}
    sortie = {"protocole": {"forces": FORCES, "profondeurs": PROFONDEURS, "graines": GRAINES, "n": N,
                            "prevalence_cible": PREVALENCE_CIBLE, "learning_rate": 0.08, "n_estimators_fixe": 180},
              "resultats": res, "verdicts": verdicts(res), "duree_s": round(time.time() - t0, 1)}
    SORTIE.write_text(json.dumps(sortie, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v["verdict"] if isinstance(v, dict) else v for k, v in sortie["verdicts"].items()}, indent=2, ensure_ascii=False))
    print(f"Écrit : {SORTIE}")


if __name__ == "__main__":
    main()
