# ATHÉNA - models/train_injury_model.py
# Génère un jeu de données synthétique physiologiquement plausible et entraîne
# un modèle Gradient Boosting pour prédire :
#   - la probabilité de blessure (classification binaire -> proba = Injury Risk %)
#   - un score de fatigue continu (régression, 0-10)
#
# Note méthodologique :
# En l'absence de dataset terrain labellisé (capteurs non déployés), le jeu
# d'entraînement est généré à partir de relations issues de la littérature
# sportive (ACWR / Gabbett et al., dette de sommeil, charge chronique, historique
# de blessures) avec bruit stochastique. Le modèle apprend donc à retrouver et
# généraliser ces relations non-linéaires plutôt que du pur bruit — il est
# conçu pour être reconnecté à des données réelles de capteurs sans changer
# l'architecture (mêmes features en entrée).

import json
from pathlib import Path

import numpy as np
import joblib
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, mean_absolute_error, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "acwr",                 # ratio charge aiguë/chronique
    "charge_chronique",     # charge moyenne 28j
    "fc_repos_estimee",     # fc de repos estimée
    "sommeil_h",            # heures de sommeil moyennes
    "age",
    "imc",                  # indice de masse corporelle
    "historique_blessures", # nb de blessures sur les 12 derniers mois
    "deficit_calorique_pct",# déficit calorique vs TDEE (%) : POSITIF = mange moins que le besoin
    "lactate",
    "intensite_moyenne_7j", # intensité d'effort moyenne (0-1) sur 7 jours
]

# Valeur utilisée quand une variable n'est pas mesurée : la valeur TYPIQUE du jeu d'entraînement
# (médianes arrondies, n = 8000), et non le milieu du domaine, qui plaçait par exemple
# 5 blessures passées (moyenne réelle 0,7) et une ACWR de 1,4 (zone de prudence) chez tout le monde.
# ACWR neutre (1,0) et aucune blessure passée : on n'invente pas un facteur de risque non mesuré.
VALEURS_TYPIQUES = {
    "acwr": 1.0, "charge_chronique": 420.0, "fc_repos_estimee": 62.0, "sommeil_h": 7.0,
    "age": 23.0, "imc": 22.5, "historique_blessures": 0.0, "deficit_calorique_pct": 0.0,
    "lactate": 2.2, "intensite_moyenne_7j": 0.55,
}


def _generer_dataset(n: int = 6000, graine: int = 42):
    rng = np.random.default_rng(graine)

    acwr = np.clip(rng.normal(1.05, 0.35, n), 0.3, 2.5)
    charge_chronique = np.clip(rng.normal(420, 130, n), 100, 900)
    fc_repos = np.clip(rng.normal(62, 9, n), 42, 100)
    sommeil_h = np.clip(rng.normal(7.0, 1.1, n), 3.5, 10.5)
    age = np.clip(rng.normal(24, 5, n), 14, 40).astype(int)
    imc = np.clip(rng.normal(22.5, 2.3, n), 16, 32)
    historique_blessures = rng.poisson(0.7, n)
    deficit_calorique_pct = np.clip(rng.normal(0, 15, n), -45, 45)
    lactate = np.clip(rng.normal(2.2, 1.3, n), 0.5, 12)
    intensite_moyenne_7j = np.clip(rng.normal(0.55, 0.2, n), 0.05, 1.0)

    # --- Score de risque latent (relations non-linéaires inspirées littérature) ---
    risque = np.zeros(n)
    risque += 2.6 * np.clip(acwr - 1.3, 0, None) ** 1.4          # zone ACWR dangereuse
    risque += 1.1 * np.clip(0.8 - acwr, 0, None)                  # sous-charge -> vulnérabilité
    risque += 0.9 * np.clip(7.0 - sommeil_h, 0, None)             # dette de sommeil
    risque += 0.55 * historique_blessures                          # récidive
    risque += 0.03 * np.clip(deficit_calorique_pct, 0, None)      # déficit calorique
    risque += 0.35 * np.clip(intensite_moyenne_7j - 0.7, 0, None) * 10
    risque += 0.06 * np.clip(lactate - 4, 0, None)
    risque += 0.02 * np.clip(age - 30, 0, None)
    risque += 0.04 * np.clip(np.abs(imc - 22), 0, None)
    risque += rng.normal(0, 0.9, n)  # bruit / facteurs non observés

    proba_blessure = 1 / (1 + np.exp(-(risque - 3.0)))
    y_blessure = rng.binomial(1, proba_blessure)

    # --- Score de fatigue (0-10), corrélé mais pas identique au risque de blessure ---
    fatigue = (
        3.0
        + 2.2 * np.clip(acwr - 1.1, 0, None)
        + 1.4 * np.clip(7.2 - sommeil_h, 0, None)
        + 0.5 * np.clip(intensite_moyenne_7j - 0.6, 0, None) * 10
        + 0.25 * np.clip(lactate - 3, 0, None)
        + rng.normal(0, 0.8, n)
    )
    fatigue = np.clip(fatigue, 0, 10)

    X = np.column_stack([
        acwr, charge_chronique, fc_repos, sommeil_h, age, imc,
        historique_blessures, deficit_calorique_pct, lactate, intensite_moyenne_7j,
    ])
    return X, y_blessure, fatigue


def entrainer_et_sauvegarder(recherche_hyperparametres: bool = True):
    X, y_blessure, y_fatigue = _generer_dataset(n=8000)

    X_train, X_test, yb_train, yb_test, yf_train, yf_test = train_test_split(
        X, y_blessure, y_fatigue, test_size=0.2, random_state=42
    )

    # Pondération équilibrée : la classe "blessure" est minoritaire dans le dataset
    # synthétique (~20%). Sans cela, le modèle est prudent et rate des cas positifs
    # (recall faible). Le poids par échantillon corrige ce biais sans dupliquer de données.
    poids_train = compute_sample_weight(class_weight="balanced", y=yb_train)

    if recherche_hyperparametres:
        # Recherche aléatoire (5-fold CV, scoring=ROC-AUC) au lieu d'hyperparamètres
        # fixés à la main : on laisse la validation croisée arbitrer profondeur,
        # nombre d'arbres, learning rate et sous-échantillonnage.
        espace_clf = {
            "n_estimators": randint(100, 400),
            "max_depth": randint(2, 5),
            "learning_rate": uniform(0.02, 0.18),
            "subsample": uniform(0.7, 0.3),
            "min_samples_leaf": randint(5, 40),
        }
        recherche_clf = RandomizedSearchCV(
            GradientBoostingClassifier(random_state=42, n_iter_no_change=10, validation_fraction=0.15),
            espace_clf, n_iter=25, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1,
        )
        recherche_clf.fit(X_train, yb_train, sample_weight=poids_train)
        clf = recherche_clf.best_estimator_
        params_clf = recherche_clf.best_params_

        espace_reg = {
            "n_estimators": randint(100, 400),
            "max_depth": randint(2, 5),
            "learning_rate": uniform(0.02, 0.18),
            "subsample": uniform(0.7, 0.3),
            "min_samples_leaf": randint(5, 40),
        }
        recherche_reg = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42, n_iter_no_change=10, validation_fraction=0.15),
            espace_reg, n_iter=25, cv=5, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
        )
        recherche_reg.fit(X_train, yf_train)
        reg = recherche_reg.best_estimator_
        params_reg = recherche_reg.best_params_
    else:
        clf = GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.08, random_state=42)
        clf.fit(X_train, yb_train, sample_weight=poids_train)
        params_clf = {"n_estimators": 180, "max_depth": 3, "learning_rate": 0.08}

        reg = GradientBoostingRegressor(n_estimators=180, max_depth=3, learning_rate=0.08, random_state=42)
        reg.fit(X_train, yf_train)
        params_reg = {"n_estimators": 180, "max_depth": 3, "learning_rate": 0.08}

    auc = roc_auc_score(yb_test, clf.predict_proba(X_test)[:, 1])
    mae = mean_absolute_error(yf_test, reg.predict(X_test))

    # Feature importances extraites AVANT calibration (CalibratedClassifierCV
    # enveloppe l'estimateur et n'expose pas .feature_importances_) et stockées
    # dans les métadonnées : l'explicabilité du dashboard ne dépend plus de la
    # structure interne de l'objet modèle sauvegardé.
    feature_importances = [round(float(v), 4) for v in clf.feature_importances_]

    # Calibration isotonique des probabilités : compute_sample_weight("balanced")
    # corrige le déséquilibre de classes pour l'entraînement mais déforme les
    # probabilités brutes du Gradient Boosting (elles ne reflètent plus la vraie
    # fréquence de blessure). On recalibre sur les données de validation pour que
    # l'Injury Risk (%) affiché soit une probabilité fiable, pas juste un score relatif.
    clf_calibre = CalibratedClassifierCV(estimator=clf, method="isotonic", cv=5)
    clf_calibre.fit(X_train, yb_train, sample_weight=poids_train)

    brier_avant = brier_score_loss(yb_test, clf.predict_proba(X_test)[:, 1])
    brier_apres = brier_score_loss(yb_test, clf_calibre.predict_proba(X_test)[:, 1])
    auc_calibre = roc_auc_score(yb_test, clf_calibre.predict_proba(X_test)[:, 1])

    joblib.dump(clf_calibre, ARTIFACTS_DIR / "injury_risk_model.joblib")
    joblib.dump(reg, ARTIFACTS_DIR / "fatigue_model.joblib")

    metadata = {
        "feature_names": FEATURE_NAMES,
        "feature_importances": feature_importances,
        "metrics": {
            "injury_model_auc": round(float(auc_calibre), 4),
            "fatigue_model_mae": round(float(mae), 4),
            "brier_score_avant_calibration": round(float(brier_avant), 4),
            "brier_score_apres_calibration": round(float(brier_apres), 4),
        },
        "n_train_samples": int(len(X_train)),
        "model_type": "GradientBoosting (scikit-learn, sample_weight='balanced')",
        "hyperparametres": {
            "injury_risk_model": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params_clf.items()},
            "fatigue_model": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params_reg.items()},
            "recherche": "RandomizedSearchCV, 5-fold, 25 tirages" if recherche_hyperparametres else "fixes",
        },
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Modèle Injury Risk entraîné + calibré — AUC = {auc_calibre:.3f} | Brier {brier_avant:.3f} -> {brier_apres:.3f}")
    print(f"Modèle Fatigue entraîné — MAE test = {mae:.3f}")
    return metadata


if __name__ == "__main__":
    entrainer_et_sauvegarder()
