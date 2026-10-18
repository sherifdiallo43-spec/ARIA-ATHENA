# ATHÉNA - models/analyse_avancee.py
# Analyses avancées du modèle de production (2e étape d'amélioration) :
#   1. Importance par permutation (plus fiable que l'importance native des arbres)
#   2. Courbe d'apprentissage (test empirique de l'hypothèse "le dataset est le
#      facteur limitant")
#   3. Robustesse : dégradation sous bruit et sous valeurs manquantes
#      (Gradient Boosting + imputation vs HistGradientBoosting natif NaN)
#   4. Test sur un jeu de données inédit (graine jamais utilisée à l'entraînement)
#   5. Analyse d'erreurs (faux négatifs vs vrais positifs)
#
# Répond aux critères de la finale : "Robustness & Adaptability" et
# "Error Analysis & Explanation". Lancer : python -m models.analyse_avancee

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from .train_injury_model import _generer_dataset, FEATURE_NAMES, ARTIFACTS_DIR
from .injury_risk import DOMAINE_ENTRAINEMENT, LABELS_LISIBLES, _charger_modeles, corriger_prevalence

RESULTATS_AVANCES_PATH = ARTIFACTS_DIR / "advanced_analysis.json"
FIGURES_DIR = ARTIFACTS_DIR.parent.parent / "reports" / "artifacts_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GRAINE_INEDITE = 123  # jamais utilisée pour entraîner (entraînement : 42)
MILIEUX = np.array([(DOMAINE_ENTRAINEMENT[n][0] + DOMAINE_ENTRAINEMENT[n][1]) / 2 for n in FEATURE_NAMES])


def _gb():
    return GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.08, random_state=42)


def _hgb():
    return HistGradientBoostingClassifier(max_iter=180, max_depth=4, learning_rate=0.08, random_state=42)


def _entrainer(modele, X, y):
    modele.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return modele


def _importance_permutation(gb, X_te, y_te):
    res = permutation_importance(gb, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=42)
    natif = gb.feature_importances_
    lignes = []
    for i, nom in enumerate(FEATURE_NAMES):
        lignes.append({
            "feature": nom,
            "label": LABELS_LISIBLES.get(nom, nom),
            "baisse_auc_moyenne": round(float(res.importances_mean[i]), 4),
            "ecart_type": round(float(res.importances_std[i]), 4),
            "importance_native": round(float(natif[i]), 4),
        })
    lignes.sort(key=lambda r: r["baisse_auc_moyenne"], reverse=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor="#04070a")
    ax.set_facecolor("#04070a")
    ordre = lignes[::-1]
    ax.barh([r["label"] for r in ordre], [r["baisse_auc_moyenne"] for r in ordre],
            xerr=[r["ecart_type"] for r in ordre], color="#00F5D4", ecolor="#E6F7FF")
    ax.set_xlabel("Baisse de ROC-AUC si la variable est permutée", color="#E6F7FF")
    ax.set_title("Importance par permutation (jeu de test)", color="#00F5D4")
    ax.tick_params(colors="#E6F7FF", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#2a3f47")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "permutation_importance.png", dpi=150)
    plt.close(fig)
    return lignes


def _courbe_apprentissage(X, y, n_graines=5):
    """AUC test en fonction de la taille d'entraînement (moyenne ± é.-t. sur graines)."""
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    sortie = []
    for frac in fractions:
        aucs = []
        for seed in range(n_graines):
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
            n = max(50, int(len(X_tr) * frac))
            gb = _entrainer(_gb(), X_tr[:n], y_tr[:n])
            aucs.append(roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1]))
        sortie.append({"n_train": int(max(50, int(len(X) * 0.8 * frac))),
                       "auc_moyen": round(float(np.mean(aucs)), 4),
                       "ecart_type": round(float(np.std(aucs)), 4)})

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#04070a")
    ax.set_facecolor("#04070a")
    xs = [r["n_train"] for r in sortie]
    m = np.array([r["auc_moyen"] for r in sortie]); e = np.array([r["ecart_type"] for r in sortie])
    ax.plot(xs, m, color="#00F5D4", marker="o")
    ax.fill_between(xs, m - e, m + e, color="#00F5D4", alpha=0.2)
    ax.set_xlabel("Taille de l'échantillon d'entraînement", color="#E6F7FF")
    ax.set_ylabel("ROC-AUC (test)", color="#E6F7FF")
    ax.set_title("Courbe d'apprentissage — Gradient Boosting", color="#00F5D4")
    ax.tick_params(colors="#E6F7FF")
    for s in ax.spines.values():
        s.set_color("#2a3f47")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)
    return sortie


def _imputer_milieu(X):
    Xc = X.copy()
    masque = np.isnan(Xc)
    Xc[masque] = np.take(MILIEUX, np.where(masque)[1])
    return Xc


def _robustesse(gb, hgb, X_te, y_te, n_rep=5):
    prod, _, _ = _charger_modeles()  # modèle de production (calibré) + imputation
    """Dégradation de l'AUC sous bruit gaussien et sous valeurs manquantes aléatoires."""
    sigma = X_te.std(axis=0)
    res = {"bruit": [], "manquants": []}
    for niveau in [0.0, 0.1, 0.25, 0.5, 1.0]:
        aucs = []
        for r in range(n_rep):
            rng = np.random.default_rng(r)
            Xb = X_te + rng.normal(0, 1, X_te.shape) * sigma * niveau
            aucs.append(roc_auc_score(y_te, gb.predict_proba(Xb)[:, 1]))
        res["bruit"].append({"niveau_sigma": niveau, "auc_moyen": round(float(np.mean(aucs)), 4),
                             "ecart_type": round(float(np.std(aucs)), 4)})
    for taux in [0.0, 0.1, 0.2, 0.3, 0.5]:
        a_gb, a_hgb, a_prod = [], [], []
        for r in range(n_rep):
            rng = np.random.default_rng(r)
            Xm = X_te.copy()
            Xm[rng.random(Xm.shape) < taux] = np.nan
            a_gb.append(roc_auc_score(y_te, gb.predict_proba(_imputer_milieu(Xm))[:, 1]))
            a_hgb.append(roc_auc_score(y_te, hgb.predict_proba(Xm)[:, 1]))
            a_prod.append(roc_auc_score(y_te, prod.predict_proba(_imputer_milieu(Xm))[:, 1]))
        res["manquants"].append({
            "taux_manquant": taux,
            "gb_imputation_milieu": round(float(np.mean(a_gb)), 4),
            "production_imputation_milieu": round(float(np.mean(a_prod)), 4),
            "hgb_natif_nan": round(float(np.mean(a_hgb)), 4),
        })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#04070a")
    for ax in axes:
        ax.set_facecolor("#04070a"); ax.tick_params(colors="#E6F7FF")
        for s in ax.spines.values():
            s.set_color("#2a3f47")
    axes[0].errorbar([r["niveau_sigma"] for r in res["bruit"]], [r["auc_moyen"] for r in res["bruit"]],
                     yerr=[r["ecart_type"] for r in res["bruit"]], color="#00F5D4", marker="o")
    axes[0].set_xlabel("Bruit ajouté (× écart-type de chaque variable)", color="#E6F7FF")
    axes[0].set_ylabel("ROC-AUC", color="#E6F7FF")
    axes[0].set_title("Robustesse au bruit", color="#00F5D4")
    axes[1].plot([r["taux_manquant"] for r in res["manquants"]], [r["gb_imputation_milieu"] for r in res["manquants"]],
                 color="#00F5D4", marker="o", label="GB étude + imputation")
    axes[1].plot([r["taux_manquant"] for r in res["manquants"]], [r["production_imputation_milieu"] for r in res["manquants"]],
                 color="#FFB703", marker="^", label="Modèle de production + imputation")
    axes[1].plot([r["taux_manquant"] for r in res["manquants"]], [r["hgb_natif_nan"] for r in res["manquants"]],
                 color="#9B5DE5", marker="s", label="HistGB natif NaN")
    axes[1].set_xlabel("Part de valeurs manquantes", color="#E6F7FF")
    axes[1].set_ylabel("ROC-AUC", color="#E6F7FF")
    axes[1].set_title("Robustesse aux valeurs manquantes", color="#00F5D4")
    axes[1].legend(fontsize=8, facecolor="#0b1a24", labelcolor="#E6F7FF")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_curves.png", dpi=150)
    plt.close(fig)
    return res


def _test_inedit(gb, hgb):
    """Évaluation sur un jeu généré avec une graine jamais utilisée à l'entraînement."""
    X, y, _ = _generer_dataset(n=3000, graine=GRAINE_INEDITE)
    out = {"graine": GRAINE_INEDITE, "n": int(len(X)), "taux_positifs": round(float(y.mean()), 3)}
    for nom, m in [("Gradient Boosting (production)", gb), ("HistGradientBoosting", hgb)]:
        proba = m.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out[nom] = {"roc_auc": round(float(roc_auc_score(y, proba)), 4),
                    "recall": round(float(recall_score(y, pred)), 4),
                    "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
                    "f1": round(float(f1_score(y, pred)), 4)}
    return out


def _calibration_production():
    """Fiabilité des scores du modèle de production sur données inédites."""
    from sklearn.metrics import brier_score_loss
    X, y, _ = _generer_dataset(n=3000, graine=GRAINE_INEDITE)
    prod, _, _ = _charger_modeles()
    p = prod.predict_proba(X)[:, 1]
    pc = np.array([corriger_prevalence(v) for v in p])
    bornes = [0, 0.2, 0.35, 0.5, 0.65, 1.01]
    classes = []
    for lo, hi in zip(bornes[:-1], bornes[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            classes.append({"intervalle_score": f"{lo:.2f}-{min(hi,1):.2f}", "n": int(m.sum()),
                            "score_moyen": round(float(p[m].mean()), 3),
                            "frequence_observee": round(float(y[m].mean()), 3),
                            "prob_corrigee_moyenne": round(float(pc[m].mean()), 3)})
    return {"frequence_reelle_blessure": round(float(y.mean()), 3),
            "score_moyen_production": round(float(p.mean()), 3),
            "prob_moyenne_apres_correction_prevalence": round(float(pc.mean()), 3),
            "brier_score_brut": round(float(brier_score_loss(y, p)), 4),
            "brier_apres_correction_prevalence": round(float(brier_score_loss(y, pc)), 4),
            "brier_reference_constante": round(float(y.mean() * (1 - y.mean())), 4),
            "fiabilite_par_classe": classes}


def _figure_calibration():
    """Courbe de fiabilité (scores prédits vs fréquence observée) du modèle de production,
    avant et après correction de prévalence, sur données inédites."""
    X, y, _ = _generer_dataset(n=6000, graine=GRAINE_INEDITE)
    prod, _, _ = _charger_modeles()
    p = prod.predict_proba(X)[:, 1]
    pc = np.array([corriger_prevalence(v) for v in p])
    bornes = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="Calibration parfaite")
    for scores, nom, couleur in [(p, "Score brut", "#1B6E8C"), (pc, "Après correction de prévalence", "#E67E22")]:
        xs, ys = [], []
        for lo, hi in zip(bornes[:-1], bornes[1:]):
            m = (scores >= lo) & (scores < hi + (1e-9 if hi >= 1 else 0))
            if m.sum() >= 30:   # on ne trace pas une classe de moins de 30 observations
                xs.append(scores[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, "o-", color=couleur, label=nom)
    ax.set_xlabel("Probabilité prédite"); ax.set_ylabel("Fréquence observée de blessure")
    ax.set_title("Courbe de fiabilité (données simulées inédites)"); ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=150); plt.close(fig)
    return "calibration_curve.png"


def _sous_groupes(n_bootstrap=500, graine=7):
    """AUC du modèle de production par sous-groupe (âge, historique de blessures, sommeil), IC95 bootstrap.

    ATTENTION : calculé sur des données SIMULÉES. Cela vérifie que le modèle ne se comporte pas
    très différemment d'un groupe à l'autre *dans la simulation* ; ce n'est PAS une preuve d'équité
    en conditions réelles (à refaire sur les données réelles quand l'effectif le permet)."""
    X, y, _ = _generer_dataset(n=6000, graine=GRAINE_INEDITE)
    prod, _, _ = _charger_modeles()
    p = prod.predict_proba(X)[:, 1]
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    rng = np.random.default_rng(graine)

    definitions = {
        "Âge": [("< 22 ans", X[:, col["age"]] < 22),
                ("22–28 ans", (X[:, col["age"]] >= 22) & (X[:, col["age"]] <= 28)),
                ("> 28 ans", X[:, col["age"]] > 28)],
        "Historique de blessures": [("Aucune blessure passée", X[:, col["historique_blessures"]] == 0),
                                    ("≥ 1 blessure passée", X[:, col["historique_blessures"]] >= 1)],
        "Sommeil moyen": [("< 6,5 h", X[:, col["sommeil_h"]] < 6.5), ("≥ 6,5 h", X[:, col["sommeil_h"]] >= 6.5)],
    }
    sortie = {}
    for critere, groupes in definitions.items():
        lignes = []
        for nom, masque in groupes:
            yg, pg = y[masque], p[masque]
            ligne = {"groupe": nom, "n": int(masque.sum()), "n_positifs": int(yg.sum())}
            if yg.sum() < 30 or (yg == 0).sum() < 30:
                ligne.update(auc=None, ic95=None, fiable=False)   # trop peu d'événements pour conclure
            else:
                aucs = []
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(yg), len(yg))
                    if yg[idx].min() != yg[idx].max():
                        aucs.append(roc_auc_score(yg[idx], pg[idx]))
                ligne.update(auc=round(float(roc_auc_score(yg, pg)), 3),
                             ic95=[round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)],
                             fiable=True)
            lignes.append(ligne)
        sortie[critere] = lignes
    return sortie


def _analyse_erreurs(gb, X_te, y_te):
    """Profil moyen des faux négatifs (blessures manquées) vs vrais positifs détectés."""
    pred = gb.predict(X_te)
    fn = (y_te == 1) & (pred == 0)
    tp = (y_te == 1) & (pred == 1)
    lignes = []
    for i, nom in enumerate(FEATURE_NAMES):
        lignes.append({"feature": nom, "label": LABELS_LISIBLES.get(nom, nom),
                       "moyenne_faux_negatifs": round(float(X_te[fn, i].mean()), 2) if fn.any() else None,
                       "moyenne_vrais_positifs": round(float(X_te[tp, i].mean()), 2) if tp.any() else None})
    return {"n_faux_negatifs": int(fn.sum()), "n_vrais_positifs": int(tp.sum()),
            "part_blessures_manquees": round(float(fn.sum() / max(1, (y_te == 1).sum())), 3),
            "profil": lignes}


def executer_analyse_avancee() -> dict:
    X, y, _ = _generer_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    gb = _entrainer(_gb(), X_tr, y_tr)
    hgb = _entrainer(_hgb(), X_tr, y_tr)

    resultats = {
        "importance_permutation": _importance_permutation(gb, X_te, y_te),
        "courbe_apprentissage": _courbe_apprentissage(X, y),
        "robustesse": _robustesse(gb, hgb, X_te, y_te),
        "test_donnees_inedites": _test_inedit(gb, hgb),
        "calibration_production": _calibration_production(),
        "figure_calibration": _figure_calibration(),
        "sous_groupes": _sous_groupes(),
        "analyse_erreurs": _analyse_erreurs(gb, X_te, y_te),
    }
    RESULTATS_AVANCES_PATH.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
    return resultats


if __name__ == "__main__":
    r = executer_analyse_avancee()
    print("\n=== IMPORTANCE PAR PERMUTATION (baisse d'AUC) ===")
    for l in r["importance_permutation"]:
        print(f"{l['label']:<42} -{l['baisse_auc_moyenne']:.4f} ± {l['ecart_type']:.4f}   (native={l['importance_native']:.3f})")
    print("\n=== COURBE D'APPRENTISSAGE ===")
    for l in r["courbe_apprentissage"]:
        print(f"n_train={l['n_train']:<5} AUC={l['auc_moyen']:.3f} ± {l['ecart_type']:.3f}")
    print("\n=== ROBUSTESSE AU BRUIT ===")
    for l in r["robustesse"]["bruit"]:
        print(f"bruit={l['niveau_sigma']:<4} AUC={l['auc_moyen']:.3f} ± {l['ecart_type']:.3f}")
    print("\n=== ROBUSTESSE AUX MANQUANTS ===")
    for l in r["robustesse"]["manquants"]:
        print(f"manquants={l['taux_manquant']:<4} GB étude+imput.={l['gb_imputation_milieu']:.3f}  production+imput.={l['production_imputation_milieu']:.3f}  HistGB natif={l['hgb_natif_nan']:.3f}")
    t = r["test_donnees_inedites"]
    print(f"\n=== DONNÉES INÉDITES (graine {t['graine']}, n={t['n']}) ===")
    for k in ("Gradient Boosting (production)", "HistGradientBoosting"):
        print(f"{k:<32} {t[k]}")
    c = r["calibration_production"]
    print(f"\n=== CALIBRATION (données inédites) === fréquence réelle={c['frequence_reelle_blessure']}  score moyen={c['score_moyen_production']}  après correction={c['prob_moyenne_apres_correction_prevalence']}")
    print(f"Brier brut={c['brier_score_brut']}  corrigé={c['brier_apres_correction_prevalence']}  référence constante={c['brier_reference_constante']}")
    for l in c["fiabilite_par_classe"]:
        print(l)
    print("\n=== SOUS-GROUPES (données simulées — ne prouve pas l'équité réelle) ===")
    for critere, lignes in r["sous_groupes"].items():
        for l in lignes:
            res = f"AUC={l['auc']} IC95={l['ic95']}" if l["fiable"] else "effectif insuffisant"
            print(f"{critere:<24} {l['groupe']:<24} n={l['n']:<5} positifs={l['n_positifs']:<4} {res}")
    e = r["analyse_erreurs"]
    print(f"\n=== ERREURS === blessures manquées : {e['n_faux_negatifs']} ({e['part_blessures_manquees']*100:.0f} %) vs détectées : {e['n_vrais_positifs']}")
    for l in e["profil"]:
        print(f"{l['label']:<42} FN={l['moyenne_faux_negatifs']}  TP={l['moyenne_vrais_positifs']}")
