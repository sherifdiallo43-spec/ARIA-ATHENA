# ATHÉNA - models/experiment_profondeur.py
# Expérience E2 : pourquoi l'avantage du Gradient Boosting sur la régression logistique est-il faible
# (expérience E1 non concluante) ? On fait varier UNE variable — la profondeur max des arbres, c'est-à-dire
# l'ordre maximal d'interaction entre variables — et on compare à un « oracle » : le classement par le
# risque latent du générateur SANS bruit, optimal au sens de l'AUC (la probabilité de blessure sachant X
# est monotone en ce risque). L'oracle n'existe que parce que les données sont synthétiques.
#
# Hypothèses et règles de décision : docs/E2_preenregistrement.md (écrit AVANT la première exécution).
#
# Protocoles
#   P2 (primaire) : 10 jeux générés indépendamment (graines 1000..1009, n = 6000), partition 80/20.
#   P1 (cohérence avec E1) : le jeu de E1 (graine 42, n = 6000) et ses 10 partitions 80/20 (random_state 0..9).
#                    Les partitions d'un même jeu se recouvrent : P1 ne sert pas à décider.
#
# Lancement : python -m models.experiment_profondeur   (depuis la racine du projet ; ~2,5 min)
# Sortie    : models/artifacts/experiment_profondeur.json

import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr, t as loi_t
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .train_injury_model import _generer_dataset, FEATURE_NAMES, ARTIFACTS_DIR

SORTIE = ARTIFACTS_DIR / "experiment_profondeur.json"
PROFONDEURS = [1, 2, 3, 4, 6]
GRAINES_P2 = list(range(1000, 1010))
GRAINES_P1 = list(range(10))
N = 6000


# ------------------------------------------------------------------ générateur avec vérité terrain ----

def risque_latent(X: np.ndarray) -> np.ndarray:
    """Risque latent SANS bruit, recalculé à partir des 10 variables (ordre de FEATURE_NAMES).
    Doit rester identique aux termes de train_injury_model._generer_dataset (vérifié par assertion)."""
    acwr, sommeil, hist = X[:, 0], X[:, 3], X[:, 6]
    age, imc, deficit, lactate, intensite = X[:, 4], X[:, 5], X[:, 7], X[:, 8], X[:, 9]
    r = np.zeros(len(X))
    r += 2.6 * np.clip(acwr - 1.3, 0, None) ** 1.4
    r += 1.1 * np.clip(0.8 - acwr, 0, None)
    r += 0.9 * np.clip(7.0 - sommeil, 0, None)
    r += 0.55 * hist
    r += 0.03 * np.clip(deficit, 0, None)
    r += 0.35 * np.clip(intensite - 0.7, 0, None) * 10
    r += 0.06 * np.clip(lactate - 4, 0, None)
    r += 0.02 * np.clip(age - 30, 0, None)
    r += 0.04 * np.clip(np.abs(imc - 22), 0, None)
    return r


def fatigue_latente(X: np.ndarray) -> np.ndarray:
    """Partie déterministe (avant bruit et écrêtage) du score de fatigue du générateur."""
    acwr, sommeil, lactate, intensite = X[:, 0], X[:, 3], X[:, 8], X[:, 9]
    return (3.0 + 2.2 * np.clip(acwr - 1.1, 0, None) + 1.4 * np.clip(7.2 - sommeil, 0, None)
            + 0.5 * np.clip(intensite - 0.6, 0, None) * 10 + 0.25 * np.clip(lactate - 3, 0, None))


def generer_jeu(n: int, graine: int):
    """Même jeu que _generer_dataset(n, graine) — vérifié — mais on conserve le risque latent propre."""
    X, y, fatigue = _generer_dataset(n=n, graine=graine)
    return X, y, fatigue, risque_latent(X)


def verifier_generateur():
    """Garde-fou : le risque latent recalculé doit reproduire exactement la probabilité de blessure du
    générateur (on rejoue le tirage du bruit avec la même graine)."""
    X, y, fat = _generer_dataset(n=N, graine=42)
    rng = np.random.default_rng(42)
    for loc, sc in [(1.05, 0.35), (420, 130), (62, 9), (7.0, 1.1), (24, 5), (22.5, 2.3)]:
        rng.normal(loc, sc, N)
    rng.poisson(0.7, N)
    for loc, sc in [(0, 15), (2.2, 1.3), (0.55, 0.2)]:
        rng.normal(loc, sc, N)
    bruit_r = rng.normal(0, 0.9, N)
    y_rejoue = rng.binomial(1, 1 / (1 + np.exp(-(risque_latent(X) + bruit_r - 3.0))))
    bruit_f = rng.normal(0, 0.8, N)
    fat_rejouee = np.clip(fatigue_latente(X) + bruit_f, 0, 10)
    assert np.array_equal(y_rejoue, y), "Le risque latent ne reproduit pas les étiquettes du générateur."
    assert np.allclose(fat_rejouee, fat), "La fatigue latente ne reproduit pas la fatigue du générateur."


# ------------------------------------------------------------------------------- un réplicat ----

def _gb(profondeur: int, graine: int, arret_precoce: bool):
    kw = dict(max_depth=profondeur, learning_rate=0.08, random_state=graine)
    if arret_precoce:
        return GradientBoostingClassifier(n_estimators=600, n_iter_no_change=10, validation_fraction=0.15, **kw)
    return GradientBoostingClassifier(n_estimators=180, **kw)


def evaluer_replicat(X, y, r_propre, graine_partition: int, graine_modele: int) -> dict:
    X_tr, X_te, y_tr, y_te, _, r_te = train_test_split(X, y, r_propre, test_size=0.2, random_state=graine_partition)
    poids = compute_sample_weight(class_weight="balanced", y=y_tr)
    sortie = {"oracle": float(roc_auc_score(y_te, r_te)), "n_test": int(len(y_te)),
              "taux_positifs_test": float(y_te.mean()), "gb_fixe": {}, "gb_arret": {}, "n_arbres_arret": {}}

    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000, random_state=graine_modele)
    lr.fit(sc.transform(X_tr), y_tr, sample_weight=poids)
    sortie["lr"] = float(roc_auc_score(y_te, lr.predict_proba(sc.transform(X_te))[:, 1]))

    for d in PROFONDEURS:
        m = _gb(d, graine_modele, False).fit(X_tr, y_tr, sample_weight=poids)
        sortie["gb_fixe"][str(d)] = float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
        m2 = _gb(d, graine_modele, True).fit(X_tr, y_tr, sample_weight=poids)
        sortie["gb_arret"][str(d)] = float(roc_auc_score(y_te, m2.predict_proba(X_te)[:, 1]))
        sortie["n_arbres_arret"][str(d)] = int(m2.n_estimators_)
    return sortie


def _replicat_p2(graine_jeu: int) -> dict:
    X, y, _, r = generer_jeu(N, graine_jeu)
    return evaluer_replicat(X, y, r, graine_partition=0, graine_modele=0)


def _replicat_p1(graine: int) -> dict:
    X, y, _, r = generer_jeu(N, 42)
    return evaluer_replicat(X, y, r, graine_partition=graine, graine_modele=graine)


# ----------------------------------------------------------------------------------- statistiques ----

def _resume(valeurs) -> dict:
    v = np.asarray(valeurs, dtype=float)
    return {"moyenne": round(float(v.mean()), 4), "ecart_type": round(float(v.std()), 4),   # écart-type de population, comme E1
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "valeurs": [round(float(x), 4) for x in v]}


def _diff_apparie(a, b) -> dict:
    """Différence moyenne a - b sur les réplicats appariés, IC95 de Student (n - 1 ddl)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    demi = loi_t.ppf(0.975, n - 1) * se
    return {"moyenne": round(float(m), 4), "ic95": [round(float(m - demi), 4), round(float(m + demi), 4)]}


def _agreger(replicats: list) -> dict:
    out = {"n_replicats": len(replicats), "oracle": _resume([r["oracle"] for r in replicats]),
           "lr": _resume([r["lr"] for r in replicats]), "gb_fixe": {}, "gb_arret": {}, "n_arbres_arret": {},
           "taux_positifs_test": round(float(np.mean([r["taux_positifs_test"] for r in replicats])), 4)}
    for cle in ("gb_fixe", "gb_arret"):
        for d in PROFONDEURS:
            out[cle][str(d)] = _resume([r[cle][str(d)] for r in replicats])
    for d in PROFONDEURS:
        out["n_arbres_arret"][str(d)] = round(float(np.mean([r["n_arbres_arret"][str(d)] for r in replicats])), 1)
    # différences appariées
    out["diff_vs_profondeur1"] = {
        cle: {str(d): _diff_apparie([r[cle][str(d)] for r in replicats], [r[cle]["1"] for r in replicats])
              for d in PROFONDEURS if d != 1} for cle in ("gb_fixe", "gb_arret")}
    out["diff_profondeur1_vs_lr"] = {
        cle: _diff_apparie([r[cle]["1"] for r in replicats], [r["lr"] for r in replicats]) for cle in ("gb_fixe", "gb_arret")}
    out["diff_profondeur3_vs_lr"] = _diff_apparie([r["gb_fixe"]["3"] for r in replicats], [r["lr"] for r in replicats])
    best_fixe = max(PROFONDEURS, key=lambda d: out["gb_fixe"][str(d)]["moyenne"])
    best_arret = max(PROFONDEURS, key=lambda d: out["gb_arret"][str(d)]["moyenne"])
    out["meilleure_profondeur"] = {"gb_fixe": best_fixe, "gb_arret": best_arret}
    out["ecart_oracle"] = {
        "gb_fixe": _diff_apparie([r["oracle"] for r in replicats], [r["gb_fixe"][str(best_fixe)] for r in replicats]),
        "gb_arret": _diff_apparie([r["oracle"] for r in replicats], [r["gb_arret"][str(best_arret)] for r in replicats]),
        "lr": _diff_apparie([r["oracle"] for r in replicats], [r["lr"] for r in replicats]),
    }
    return out


def verdicts(p2: dict) -> dict:
    """Applique mécaniquement les règles de docs/E2_preenregistrement.md au protocole P2, condition E2a."""
    diffs = p2["diff_vs_profondeur1"]["gb_fixe"]
    bornes_hautes = {d: diffs[d]["ic95"][1] for d in diffs}
    contredite = any(diffs[d]["ic95"][0] > 0 and diffs[d]["moyenne"] >= 0.010 for d in diffs)
    if all(b < 0.010 for b in bornes_hautes.values()):
        h2 = "confirmée"
    elif contredite:
        h2 = "contredite"
    else:
        h2 = "non concluante"
    ecart = p2["ecart_oracle"]["gb_fixe"]["moyenne"]
    h3 = "confirmée" if ecart < 0.030 else "contredite"
    d1 = p2["diff_profondeur1_vs_lr"]["gb_fixe"]
    if d1["ic95"][0] > 0:
        h4 = "confirmée"
    elif d1["ic95"][1] < 0:
        h4 = "contredite"
    else:
        h4 = "non concluante"
    return {"H2": {"verdict": h2, "bornes_hautes_ic95": bornes_hautes, "seuil": 0.010},
            "H3": {"verdict": h3, "ecart_moyen_oracle_meilleure_gb": ecart, "seuil": 0.030},
            "H4": {"verdict": h4, "diff_profondeur1_vs_lr": d1}}


# ------------------------------------------------------------------------------- analyses descriptives ----

def plancher_bruit_fatigue() -> dict:
    """MAE / RMSE / R² du prédicteur qui connaît la partie déterministe de la fatigue, sur le test de E1."""
    X, _, fat = _generer_dataset(n=N, graine=42)
    _, X_te, _, f_te = train_test_split(X, fat, test_size=0.2, random_state=42)
    pred = np.clip(fatigue_latente(X_te), 0, 10)
    return {"mae": round(float(mean_absolute_error(f_te, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(f_te, pred))), 4),
            "r2": round(float(r2_score(f_te, pred)), 4)}


def oracle_split_e1() -> dict:
    """AUC de l'oracle sur le jeu de test de E1 (graine 42, partition random_state=42), pour la table de E1."""
    X, y, _ = _generer_dataset(n=N, graine=42)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    return {"auc": round(float(roc_auc_score(y_te, risque_latent(X_te))), 4), "n_test": int(len(y_te))}


def importance_vs_verite() -> dict:
    """Corrélation de rang entre l'importance par permutation (baisse d'AUC) du GB de référence de E1
    et celle de l'oracle, sur le test de E1 (graine 42)."""
    X, y, _ = _generer_dataset(n=N, graine=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    poids = compute_sample_weight(class_weight="balanced", y=y_tr)
    gb = GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.08, random_state=42)
    gb.fit(X_tr, y_tr, sample_weight=poids)
    imp = permutation_importance(gb, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=0)
    rng = np.random.default_rng(0)
    base = roc_auc_score(y_te, risque_latent(X_te))
    imp_oracle = []
    for j in range(X_te.shape[1]):
        baisses = []
        for _ in range(20):
            Xp = X_te.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            baisses.append(base - roc_auc_score(y_te, risque_latent(Xp)))
        imp_oracle.append(float(np.mean(baisses)))
    rho = spearmanr(imp.importances_mean, imp_oracle)
    return {"spearman_rho": round(float(rho.statistic), 3), "p_value": round(float(rho.pvalue), 4),
            "variables": FEATURE_NAMES,
            "baisse_auc_modele": [round(float(v), 4) for v in imp.importances_mean],
            "baisse_auc_oracle": [round(v, 4) for v in imp_oracle]}


def main():
    t0 = time.time()
    verifier_generateur()
    print("Générateur vérifié : le risque latent reproduit exactement les étiquettes (graine 42).", flush=True)

    rep_p2 = Parallel(n_jobs=-1)(delayed(_replicat_p2)(g) for g in GRAINES_P2)
    print(f"P2 terminé ({time.time() - t0:.0f} s)", flush=True)
    rep_p1 = Parallel(n_jobs=-1)(delayed(_replicat_p1)(g) for g in GRAINES_P1)
    print(f"P1 terminé ({time.time() - t0:.0f} s)", flush=True)

    p2, p1 = _agreger(rep_p2), _agreger(rep_p1)
    resultats = {
        "protocole": {
            "profondeurs": PROFONDEURS, "learning_rate": 0.08, "n_estimators_fixe": 180,
            "arret_precoce": {"n_estimators_max": 600, "n_iter_no_change": 10, "validation_fraction": 0.15},
            "ponderation": "balanced", "n": N, "graines_p2": GRAINES_P2, "partitions_p1": GRAINES_P1,
            "ecart_type": "population (ddof=0), comme E1 ; IC95 de Student sur différences appariées (ddof=1)",
        },
        "P2_jeux_independants": p2, "P1_jeu_E1_partitions": p1,
        "verdicts": verdicts(p2),
        "plancher_bruit_fatigue": plancher_bruit_fatigue(),
        "oracle_split_E1": oracle_split_e1(),
        "importance_vs_verite": importance_vs_verite(),
        "duree_s": round(time.time() - t0, 1),
    }
    SORTIE.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultats["verdicts"], indent=2, ensure_ascii=False))
    print(f"Écrit : {SORTIE}")


if __name__ == "__main__":
    main()
