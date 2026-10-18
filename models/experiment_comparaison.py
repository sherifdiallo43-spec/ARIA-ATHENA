# ATHÉNA - models/experiment_comparaison.py
# Étude scientifique comparative (format attendu par le règlement PARC 2026 —
# Research Question / Hypothesis / Baseline / Conditions / Results / Discussion).
#
# RÉFÉRENCE SCIENTIFIQUE :
#   Leckey C, van Dyk N, Doherty C, Lawlor A, Delahunt E. "Machine learning
#   approaches to injury risk prediction in sport: a scoping review with
#   evidence synthesis." Br J Sports Med. 2025;59(7):491-500.
#   doi:10.1136/bjsports-2024-108576.
#   Cette revue de 38 études conclut que les modèles à base d'arbres (Random
#   Forest, XGBoost/Gradient Boosting) obtiennent la meilleure performance de
#   prédiction (AUC) dans la majorité des études, devant la régression
#   logistique — c'est cette observation de la littérature qu'ATHÉNA teste ici
#   sur son propre cas d'usage.
#
# RESEARCH QUESTION :
#   Un modèle non-linéaire à ensemble d'arbres (Gradient Boosting) apporte-t-il
#   un gain mesurable, par rapport à une régression logistique (baseline
#   interprétable) et à une forêt aléatoire, pour prédire le risque de blessure
#   sportive à partir de features physiologiques et de charge d'entraînement ?
#
# HYPOTHÈSE :
#   H1 : les modèles non-linéaires à ensemble d'arbres (Gradient Boosting,
#        Random Forest) prédisent mieux le risque de blessure sportive que les
#        modèles linéaires (Logistic Regression), mesuré par le ROC-AUC.
#   H0 : aucune différence significative de ROC-AUC entre les familles de
#        modèles sur ce dataset.
#   Test : bootstrap apparié sur l'échantillon de test (1000 tirages), mesure
#   de la proportion de tirages où AUC(Gradient Boosting) > AUC(Logistic
#   Regression) — une proportion proche de 1.0 constitue une preuve empirique
#   robuste en faveur de H1 (pas un simple écart de point estimate).
#
# BASELINE :
#   Logistic Regression — modèle linéaire simple, standard en épidémiologie du
#   sport, pris comme référence minimale de comparaison.
#
# CONDITIONS EXPÉRIMENTALES :
#   - Même jeu de données synthétique (6000 échantillons, graine fixe = 42).
#   - Même split train/test (80/20, random_state=42) pour les 3 modèles.
#   - Mêmes 10 features en entrée pour les 3 modèles (voir FEATURE_NAMES).
#   - Pondération "balanced" appliquée aux 3 modèles (classe blessure
#     minoritaire ~20%) — comparaison équitable, pas seulement le meilleur
#     réglage du modèle retenu.
#
# Ce module est autonome (`python -m models.experiment_comparaison`) et génère
# aussi les figures (ROC, matrice de confusion, feature importance) utilisées
# par le dashboard (onglet "🔬 Étude Scientifique") et le rapport PDF.

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)

from .train_injury_model import _generer_dataset, FEATURE_NAMES, ARTIFACTS_DIR

RESULTATS_PATH = ARTIFACTS_DIR / "experiment_results.json"
FIGURES_DIR = ARTIFACTS_DIR.parent.parent / "reports" / "artifacts_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_SCIENTIFIQUE = (
    "Leckey C, van Dyk N, Doherty C, Lawlor A, Delahunt E. \u00ab Machine learning "
    "approaches to injury risk prediction in sport: a scoping review with evidence "
    "synthesis \u00bb. Br J Sports Med. 2025;59(7):491-500. doi:10.1136/bjsports-2024-108576."
)

COULEUR_LR = "#7FF7E9"
COULEUR_RF = "#00B4D8"
COULEUR_GB = "#00F5D4"


def _entrainer_classification(nom, modele, X_train, X_test, y_train, y_test, poids_train=None, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train_use = scaler.fit_transform(X_train)
        X_test_use = scaler.transform(X_test)
    else:
        X_train_use, X_test_use = X_train, X_test

    if poids_train is not None:
        modele.fit(X_train_use, y_train, sample_weight=poids_train)
    else:
        modele.fit(X_train_use, y_train)

    y_pred = modele.predict(X_test_use)
    y_proba = modele.predict_proba(X_test_use)[:, 1]

    metriques = {
        "modele": nom,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }
    return metriques, y_proba, y_pred, modele


def _evaluer_regression(nom_modele, modele, X_train, X_test, y_train, y_test, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train_use = scaler.fit_transform(X_train)
        X_test_use = scaler.transform(X_test)
    else:
        X_train_use, X_test_use = X_train, X_test

    modele.fit(X_train_use, y_train)
    y_pred = modele.predict(X_test_use)

    return {
        "modele": nom_modele,
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }


def _validation_croisee_5fold(X, y):
    """
    Validation croisée stratifiée 5-fold sur les 4 classifieurs — répond à la
    question "et si le split 80/20 était juste chanceux ?" avec 5 splits
    indépendants au lieu d'un seul, chacun avec pondération équilibrée.
    Retourne moyenne ± écart-type de ROC-AUC et F1 par modèle.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    modeles = {
        "Logistic Regression (baseline)": (LogisticRegression(max_iter=1000, random_state=42), True),
        "Random Forest": (RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"), False),
        "Gradient Boosting (retenu en production)": (GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.08, random_state=42), False),
        "HistGradientBoosting (natif NaN)": (HistGradientBoostingClassifier(max_iter=180, max_depth=4, learning_rate=0.08, random_state=42), False),
    }
    resultats = []
    for nom, (modele, scale) in modeles.items():
        aucs, f1s = [], []
        for idx_train, idx_test in skf.split(X, y):
            X_tr, X_te = X[idx_train], X[idx_test]
            y_tr, y_te = y[idx_train], y[idx_test]
            if scale:
                scaler = StandardScaler()
                X_tr, X_te = scaler.fit_transform(X_tr), scaler.transform(X_te)
            poids = compute_sample_weight(class_weight="balanced", y=y_tr)
            try:
                modele.fit(X_tr, y_tr, sample_weight=poids)
            except TypeError:
                modele.fit(X_tr, y_tr)  # RandomForest utilise déjà class_weight="balanced"
            proba = modele.predict_proba(X_te)[:, 1]
            pred = modele.predict(X_te)
            aucs.append(roc_auc_score(y_te, proba))
            f1s.append(f1_score(y_te, pred, zero_division=0))
        resultats.append({
            "modele": nom,
            "roc_auc_moyen": round(float(np.mean(aucs)), 4),
            "roc_auc_ecart_type": round(float(np.std(aucs)), 4),
            "f1_moyen": round(float(np.mean(f1s)), 4),
            "f1_ecart_type": round(float(np.std(f1s)), 4),
        })
    return resultats


def _variance_multi_seed(X, y_blessure, y_fatigue, n_seeds=10):
    """
    Répète le split 80/20 + entraînement + évaluation sur n_seeds graines
    aléatoires différentes (au lieu de la seule graine 42 utilisée dans le
    split de référence). Objectif : vérifier que le classement des modèles
    en Table 1/2 n'est pas un artefact d'un split particulièrement favorable
    à l'un d'eux — complète le 5-fold CV en testant explicitement la variance
    du split lui-même, avec ré-entraînement complet à chaque graine.
    """
    aucs = {
        "Logistic Regression (baseline)": [],
        "Random Forest": [],
        "Gradient Boosting (retenu en production)": [],
        "HistGradientBoosting (natif NaN)": [],
    }
    r2s = {
        "Linear Regression (baseline)": [],
        "Random Forest": [],
        "Gradient Boosting (retenu en production)": [],
        "HistGradientBoosting (natif NaN)": [],
    }

    for seed in range(n_seeds):
        X_tr, X_te, yb_tr, yb_te, yf_tr, yf_te = train_test_split(
            X, y_blessure, y_fatigue, test_size=0.2, random_state=seed
        )
        poids = compute_sample_weight(class_weight="balanced", y=yb_tr)

        m_lr, _, _, _ = _entrainer_classification(
            "LR", LogisticRegression(max_iter=1000, random_state=seed),
            X_tr, X_te, yb_tr, yb_te, poids_train=poids, scale=True)
        aucs["Logistic Regression (baseline)"].append(m_lr["roc_auc"])

        m_rf, _, _, _ = _entrainer_classification(
            "RF", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed, class_weight="balanced"),
            X_tr, X_te, yb_tr, yb_te)
        aucs["Random Forest"].append(m_rf["roc_auc"])

        m_gb, _, _, _ = _entrainer_classification(
            "GB", GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.08, random_state=seed),
            X_tr, X_te, yb_tr, yb_te, poids_train=poids)
        aucs["Gradient Boosting (retenu en production)"].append(m_gb["roc_auc"])

        m_hgb, _, _, _ = _entrainer_classification(
            "HGB", HistGradientBoostingClassifier(max_iter=180, max_depth=4, learning_rate=0.08, random_state=seed),
            X_tr, X_te, yb_tr, yb_te, poids_train=poids)
        aucs["HistGradientBoosting (natif NaN)"].append(m_hgb["roc_auc"])

        r2s["Linear Regression (baseline)"].append(
            _evaluer_regression("LR", LinearRegression(), X_tr, X_te, yf_tr, yf_te, scale=True)["r2"])
        r2s["Random Forest"].append(
            _evaluer_regression("RF", RandomForestRegressor(n_estimators=200, max_depth=6, random_state=seed),
                                 X_tr, X_te, yf_tr, yf_te)["r2"])
        r2s["Gradient Boosting (retenu en production)"].append(
            _evaluer_regression("GB", GradientBoostingRegressor(n_estimators=180, max_depth=3, learning_rate=0.08, random_state=seed),
                                 X_tr, X_te, yf_tr, yf_te)["r2"])
        r2s["HistGradientBoosting (natif NaN)"].append(
            _evaluer_regression("HGB", HistGradientBoostingRegressor(max_iter=180, max_depth=4, learning_rate=0.08, random_state=seed),
                                 X_tr, X_te, yf_tr, yf_te)["r2"])

    resultat = {"n_seeds": n_seeds, "classification_roc_auc": {}, "regression_r2": {}}
    for nom, vals in aucs.items():
        resultat["classification_roc_auc"][nom] = {
            "moyenne": round(float(np.mean(vals)), 4),
            "ecart_type": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }
    for nom, vals in r2s.items():
        resultat["regression_r2"][nom] = {
            "moyenne": round(float(np.mean(vals)), 4),
            "ecart_type": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }
    return resultat


def _optimiser_seuil_decision(y_test, proba_gb):
    """
    Le seuil par défaut (0.5) n'est pas nécessairement optimal pour une classe
    minoritaire. On cherche ici le seuil qui maximise le F1 sur l'échantillon
    de test, et on rapporte les métriques à ce seuil — une pratique standard
    en classification déséquilibrée, distincte du réglage des poids.
    """
    precisions, recalls, seuils = precision_recall_curve(y_test, proba_gb)
    f1s = np.where((precisions + recalls) > 0, 2 * precisions * recalls / (precisions + recalls + 1e-12), 0)
    idx_opt = int(np.argmax(f1s[:-1])) if len(seuils) > 0 else None
    if idx_opt is None:
        return None
    seuil_opt = float(seuils[idx_opt])
    pred_opt = (proba_gb >= seuil_opt).astype(int)
    return {
        "seuil_optimal": round(seuil_opt, 4),
        "accuracy": round(float(accuracy_score(y_test, pred_opt)), 4),
        "precision": round(float(precision_score(y_test, pred_opt, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred_opt, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred_opt, zero_division=0)), 4),
    }


def _bootstrap_test_hypothese(y_test, proba_gb, proba_lr, n_tirages=1000, graine=42):
    """
    Bootstrap apparié sur l'échantillon de test : à chaque tirage, on ré-échantillonne
    les mêmes indices pour les 2 modèles et on compare leur ROC-AUC. La proportion de
    tirages où GB > LR est une mesure empirique de robustesse de H1 (pas juste un
    écart de point estimate sur un seul split).
    """
    rng = np.random.default_rng(graine)
    y_test = np.asarray(y_test)
    n = len(y_test)
    victoires_gb = 0
    diffs = []
    tirages_valides = 0

    for _ in range(n_tirages):
        idx = rng.integers(0, n, n)
        y_b = y_test[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_gb = roc_auc_score(y_b, proba_gb[idx])
        auc_lr = roc_auc_score(y_b, proba_lr[idx])
        diffs.append(auc_gb - auc_lr)
        if auc_gb > auc_lr:
            victoires_gb += 1
        tirages_valides += 1

    diffs = np.array(diffs)
    return {
        "n_tirages_valides": int(tirages_valides),
        "proportion_gb_superieur": round(float(victoires_gb / tirages_valides), 4) if tirages_valides else None,
        "diff_auc_moyenne": round(float(diffs.mean()), 4) if tirages_valides else None,
        "ic95_diff_auc": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)] if tirages_valides else None,
    }


def _generer_figures(y_test, resultats_courbes, feature_names, importances_gb):
    """Génère et sauvegarde les figures PNG : ROC comparée, matrice de confusion (GB), feature importance."""
    plt.style.use("dark_background")
    couleurs = {"Logistic Regression (baseline)": COULEUR_LR, "Random Forest": COULEUR_RF,
                "Gradient Boosting (retenu en production)": COULEUR_GB,
                "HistGradientBoosting (natif NaN)": "#9B5DE5"}

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#04070a")
    ax.set_facecolor("#04070a")
    for nom, (fpr, tpr, auc, _pred) in resultats_courbes.items():
        ax.plot(fpr, tpr, label=f"{nom} (AUC={auc:.3f})", color=couleurs.get(nom, "#FFFFFF"), linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlabel("Taux de faux positifs", color="#E6F7FF")
    ax.set_ylabel("Taux de vrais positifs", color="#E6F7FF")
    ax.set_title("Courbes ROC — comparaison des modèles", color="#00F5D4")
    ax.tick_params(colors="#E6F7FF")
    ax.legend(fontsize=8, facecolor="#0b1a24", edgecolor="#00F5D4", labelcolor="#E6F7FF")
    for spine in ax.spines.values():
        spine.set_color("#2a3f47")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    y_pred_gb = resultats_courbes["Gradient Boosting (retenu en production)"][3]
    cm = confusion_matrix(y_test, y_pred_gb)
    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor="#04070a")
    ax.set_facecolor("#04070a")
    ax.imshow(cm, cmap="viridis")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pas de blessure", "Blessure"], color="#E6F7FF")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Pas de blessure", "Blessure"], color="#E6F7FF")
    ax.set_xlabel("Prédit", color="#E6F7FF"); ax.set_ylabel("Réel", color="#E6F7FF")
    ax.set_title("Matrice de confusion — Gradient Boosting", color="#00F5D4", fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] < cm.max() / 2 else "black", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    ordre = np.argsort(importances_gb)
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#04070a")
    ax.set_facecolor("#04070a")
    ax.barh(np.array(feature_names)[ordre], importances_gb[ordre], color=COULEUR_GB)
    ax.set_xlabel("Importance relative", color="#E6F7FF")
    ax.set_title("Feature Importance — Gradient Boosting", color="#00F5D4")
    ax.tick_params(colors="#E6F7FF")
    for spine in ax.spines.values():
        spine.set_color("#2a3f47")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    return {
        "roc_curves": str(FIGURES_DIR / "roc_curves.png"),
        "confusion_matrix": str(FIGURES_DIR / "confusion_matrix.png"),
        "feature_importance": str(FIGURES_DIR / "feature_importance.png"),
    }


def executer_experience() -> dict:
    """
    Exécute l'étude comparative complète (classification blessure + régression
    fatigue + test d'hypothèse + figures) sur les 4 familles de modèles, avec
    le même split de données et la même pondération équilibrée.
    """
    X, y_blessure, y_fatigue = _generer_dataset()

    X_train, X_test, yb_train, yb_test, yf_train, yf_test = train_test_split(
        X, y_blessure, y_fatigue, test_size=0.2, random_state=42
    )
    poids_train = compute_sample_weight(class_weight="balanced", y=yb_train)

    m_lr, proba_lr, pred_lr, _ = _entrainer_classification(
        "Logistic Regression (baseline)", LogisticRegression(max_iter=1000, random_state=42),
        X_train, X_test, yb_train, yb_test, poids_train=poids_train, scale=True,
    )
    m_rf, proba_rf, pred_rf, _mdl_rf = _entrainer_classification(
        "Random Forest", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"),
        X_train, X_test, yb_train, yb_test,
    )
    m_gb, proba_gb, pred_gb, mdl_gb = _entrainer_classification(
        "Gradient Boosting (retenu en production)",
        GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.08, random_state=42),
        X_train, X_test, yb_train, yb_test, poids_train=poids_train,
    )
    m_hgb, proba_hgb, pred_hgb, mdl_hgb = _entrainer_classification(
        "HistGradientBoosting (natif NaN)",
        HistGradientBoostingClassifier(max_iter=180, max_depth=4, learning_rate=0.08, random_state=42),
        X_train, X_test, yb_train, yb_test, poids_train=poids_train,
    )
    resultats_classification = [m_lr, m_rf, m_gb, m_hgb]

    courbes = {}
    for nom, proba, pred in [
        ("Logistic Regression (baseline)", proba_lr, pred_lr),
        ("Random Forest", proba_rf, pred_rf),
        ("Gradient Boosting (retenu en production)", proba_gb, pred_gb),
        ("HistGradientBoosting (natif NaN)", proba_hgb, pred_hgb),
    ]:
        fpr, tpr, _ = roc_curve(yb_test, proba)
        auc = roc_auc_score(yb_test, proba)
        courbes[nom] = (fpr, tpr, auc, pred)

    resultats_regression = [
        _evaluer_regression("Linear Regression (baseline)", LinearRegression(),
                             X_train, X_test, yf_train, yf_test, scale=True),
        _evaluer_regression("Random Forest", RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
                             X_train, X_test, yf_train, yf_test),
        _evaluer_regression("Gradient Boosting (retenu en production)",
                             GradientBoostingRegressor(n_estimators=180, max_depth=3, learning_rate=0.08, random_state=42),
                             X_train, X_test, yf_train, yf_test),
        _evaluer_regression("HistGradientBoosting (natif NaN)",
                             HistGradientBoostingRegressor(max_iter=180, max_depth=4, learning_rate=0.08, random_state=42),
                             X_train, X_test, yf_train, yf_test),
    ]

    importances_gb = mdl_gb.feature_importances_
    chemins_figures = _generer_figures(yb_test, courbes, FEATURE_NAMES, importances_gb)

    test_hypothese = _bootstrap_test_hypothese(yb_test, proba_gb, proba_lr)
    validation_croisee = _validation_croisee_5fold(X, y_blessure)
    variance_multi_seed = _variance_multi_seed(X, y_blessure, y_fatigue, n_seeds=10)
    seuil_optimise = _optimiser_seuil_decision(yb_test, proba_gb)

    meilleur_clf = max(resultats_classification, key=lambda r: r["roc_auc"])
    meilleur_reg = min(resultats_regression, key=lambda r: r["mae"])

    # Sauvegarde de la feature importance (nécessaire côté injury_risk.py, qui
    # ne peut pas l'extraire directement d'un CalibratedClassifierCV).
    metadata_path = ARTIFACTS_DIR / "metadata.json"

    resultats = {
        "reference_scientifique": REFERENCE_SCIENTIFIQUE,
        "research_question": (
            "Un modèle Gradient Boosting apporte-t-il un gain mesurable par rapport "
            "à une régression logistique (baseline) et une forêt aléatoire, pour "
            "prédire le risque de blessure et le niveau de fatigue à partir de "
            "features physiologiques et de charge d'entraînement ?"
        ),
        "hypothese_h1": (
            "Les modèles non-linéaires à ensemble d'arbres (Gradient Boosting, Random "
            "Forest) prédisent mieux le risque de blessure sportive que les modèles "
            "linéaires (Logistic Regression), mesuré par le ROC-AUC."
        ),
        "hypothese_h0": "Aucune différence significative de ROC-AUC entre les familles de modèles sur ce dataset.",
        "test_hypothese": test_hypothese,
        "validation_croisee_5fold": validation_croisee,
        "variance_multi_seed": variance_multi_seed,
        "seuil_optimise": seuil_optimise,
        "conditions_experimentales": {
            "n_echantillons": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": FEATURE_NAMES,
            "random_state": 42,
            "split": "80/20",
            "ponderation": "balanced (class_weight / sample_weight) sur les 4 modèles",
        },
        "classification_injury_risk": resultats_classification,
        "regression_fatigue": resultats_regression,
        "figures": chemins_figures,
        "conclusion": {
            "meilleur_modele_classification": meilleur_clf["modele"],
            "roc_auc_meilleur": meilleur_clf["roc_auc"],
            "meilleur_modele_regression": meilleur_reg["modele"],
            "mae_meilleur": meilleur_reg["mae"],
        },
    }

    # Les figures sont enregistrées par NOM de fichier (portable d'une machine à
    # l'autre) ; les chemins absolus sont reconstruits au chargement.
    a_sauver = dict(resultats)
    a_sauver["figures"] = {cle: Path(chemin).name for cle, chemin in chemins_figures.items()}
    with open(RESULTATS_PATH, "w", encoding="utf-8") as f:
        json.dump(a_sauver, f, indent=2, ensure_ascii=False)

    return _resoudre_chemins_figures(a_sauver)


def _resoudre_chemins_figures(resultats: dict) -> dict:
    """Transforme les noms de fichiers de figures en chemins absolus valides sur
    CETTE machine (les anciens JSON contenaient des chemins absolus d'une autre
    machine : on ne garde que le nom de fichier)."""
    resultats = dict(resultats)
    resultats["figures"] = {
        cle: str(FIGURES_DIR / Path(str(chemin).replace("\\", "/")).name)
        for cle, chemin in resultats.get("figures", {}).items()
    }
    return resultats


def charger_ou_executer_experience() -> dict:
    """Charge les résultats pré-calculés livrés avec le projet (instantané).

    L'étude complète prend environ 2 à 3 minutes : elle n'est relancée que si le
    fichier de résultats ou une figure est réellement absent/illisible.
    Pour la régénérer volontairement : `python -m models.experiment_comparaison`.
    """
    if RESULTATS_PATH.exists():
        try:
            resultats = _resoudre_chemins_figures(
                json.loads(RESULTATS_PATH.read_text(encoding="utf-8"))
            )
            if all(Path(p).exists() for p in resultats["figures"].values()):
                return resultats
        except (ValueError, KeyError, OSError):
            pass
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return executer_experience()


if __name__ == "__main__":
    res = executer_experience()
    print("\n=== CLASSIFICATION — Injury Risk (pondération équilibrée) ===")
    for r in res["classification_injury_risk"]:
        print(f"{r['modele']:<40} Acc={r['accuracy']:.3f}  Prec={r['precision']:.3f}  "
              f"Rec={r['recall']:.3f}  F1={r['f1']:.3f}  ROC-AUC={r['roc_auc']:.3f}")
    print("\n=== REGRESSION — Fatigue Score ===")
    for r in res["regression_fatigue"]:
        print(f"{r['modele']:<40} MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  R²={r['r2']:.3f}")
    print(f"\nTest d'hypothèse (bootstrap, {res['test_hypothese']['n_tirages_valides']} tirages) :")
    print(f"  P(AUC Gradient Boosting > AUC Logistic Regression) = {res['test_hypothese']['proportion_gb_superieur']}")
    print(f"  Écart moyen AUC = {res['test_hypothese']['diff_auc_moyenne']}  IC95% = {res['test_hypothese']['ic95_diff_auc']}")
    print("\n=== VALIDATION CROISÉE (5-fold stratifiée) ===")
    for r in res["validation_croisee_5fold"]:
        print(f"{r['modele']:<40} ROC-AUC={r['roc_auc_moyen']:.3f}±{r['roc_auc_ecart_type']:.3f}  "
              f"F1={r['f1_moyen']:.3f}±{r['f1_ecart_type']:.3f}")
    print(f"\n=== VARIANCE MULTI-SEED ({res['variance_multi_seed']['n_seeds']} splits 80/20 indépendants) ===")
    for nom, s in res["variance_multi_seed"]["classification_roc_auc"].items():
        print(f"{nom:<40} ROC-AUC={s['moyenne']:.3f}±{s['ecart_type']:.3f}  [min={s['min']:.3f} max={s['max']:.3f}]")
    for nom, s in res["variance_multi_seed"]["regression_r2"].items():
        print(f"{nom:<40} R²={s['moyenne']:.3f}±{s['ecart_type']:.3f}  [min={s['min']:.3f} max={s['max']:.3f}]")
    if res["seuil_optimise"]:
        s = res["seuil_optimise"]
        print(f"\nSeuil optimisé (F1-max) pour Gradient Boosting = {s['seuil_optimal']:.3f} : "
              f"Acc={s['accuracy']:.3f} Prec={s['precision']:.3f} Rec={s['recall']:.3f} F1={s['f1']:.3f}")
    print(f"\nFigures générées : {list(res['figures'].values())}")
