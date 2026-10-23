# ATHÉNA - models/injury_risk.py
# Wrapper d'inférence pour les modèles ML entraînés (Gradient Boosting).
# Fournit : Injury Risk Score (%), Fatigue Level, Overtraining Risk,
# ainsi que les facteurs contributifs principaux (feature importance).

import json
import warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
import joblib

from analytics.metrics import charge_quotidienne

from .train_injury_model import (FEATURE_NAMES, VALEURS_TYPIQUES, ARTIFACTS_DIR, entrainer_et_sauvegarder,
                                 _generer_dataset)

LABELS_LISIBLES = {
    "acwr": "Ratio charge aiguë/chronique (ACWR)",
    "charge_chronique": "Charge chronique (28j)",
    "fc_repos_estimee": "Fréquence cardiaque de repos",
    "sommeil_h": "Heures de sommeil",
    "age": "Âge",
    "imc": "IMC",
    "historique_blessures": "Historique de blessures",
    "deficit_calorique_pct": "Écart calorique vs TDEE",
    "lactate": "Taux de lactate",
    "intensite_moyenne_7j": "Intensité d'effort (7 derniers jours)",
}

DOMAINE_ENTRAINEMENT = {
    "acwr": (0.3, 2.5),
    "charge_chronique": (100, 900),
    "fc_repos_estimee": (42, 100),
    "sommeil_h": (3.5, 10.5),
    "age": (14, 40),
    "imc": (16, 32),
    "historique_blessures": (0, 10),
    "deficit_calorique_pct": (-45, 45),
    "lactate": (0.5, 12),
    "intensite_moyenne_7j": (0.05, 1.0),
}


@lru_cache(maxsize=2)
def _charger_modeles(reentrainer_si_echec: bool = True):
    """Charge les modèles depuis le disque ; les entraîne si absents (1er lancement).

    Une version de scikit-learn différente de celle qui a sauvegardé les modèles ne fait
    qu'émettre un avertissement (InconsistentVersionWarning) : il est capturé et rendu
    visible dans metadata["avertissement_version"] au lieu de passer inaperçu.
    `reentrainer_si_echec=False` (vérificateur de démo) : lève l'erreur au lieu de réentraîner en silence."""
    chemin_clf = ARTIFACTS_DIR / "injury_risk_model.joblib"
    chemin_reg = ARTIFACTS_DIR / "fatigue_model.joblib"
    chemin_meta = ARTIFACTS_DIR / "metadata.json"

    if not (chemin_clf.exists() and chemin_reg.exists()):
        if not reentrainer_si_echec:
            raise FileNotFoundError("Fichiers .joblib des modèles absents de models/artifacts.")
        entrainer_et_sauvegarder()

    with warnings.catch_warnings(record=True) as alertes:
        warnings.simplefilter("always")
        try:
            clf = joblib.load(chemin_clf)
            reg = joblib.load(chemin_reg)
        except Exception:
            if not reentrainer_si_echec:
                raise
            entrainer_et_sauvegarder()
            clf = joblib.load(chemin_clf)
            reg = joblib.load(chemin_reg)

    metadata = json.loads(chemin_meta.read_text()) if chemin_meta.exists() else {}
    versions = [str(a.message) for a in alertes if type(a.message).__name__ == "InconsistentVersionWarning"]
    if versions:
        metadata["avertissement_version"] = (
            "Modèles sauvegardés avec une autre version de scikit-learn que celle installée : les scores "
            "peuvent différer. Installer la version de requirements.txt. " + versions[0][:200])
    return clf, reg, metadata


@lru_cache(maxsize=1)
def _prevalence_entrainement() -> float:
    """Fréquence réelle de la classe 'blessure' dans les données d'entraînement (~20 %)."""
    return float(_generer_dataset(n=8000)[1].mean())   # même taille qu'à l'entraînement


def corriger_prevalence(proba: float) -> float:
    """
    Le modèle est entraîné avec une pondération équilibrée (classes ramenées à
    50/50), donc son score est calibré pour une population où la moitié des
    athlètes se blessent. Cette correction de prior (théorème de Bayes) le
    ramène à la prévalence réelle du dataset (~20 %). Le score affiché dans le
    dashboard reste le score équilibré (plus sensible, adapté à la détection) ;
    cette valeur est exposée en plus, pour qui veut une probabilité absolue.
    """
    pi = _prevalence_entrainement()
    num = proba * pi
    return float(num / (num + (1 - proba) * (1 - pi)))


def _categorie_risque(pct: float) -> dict:
    if pct < 33:
        return {"niveau": "Faible", "couleur": "#1F7A52", "badge": "🟢"}
    elif pct < 66:
        return {"niveau": "Modéré", "couleur": "#B26B00", "badge": "🟠"}
    else:
        return {"niveau": "Élevé", "couleur": "#C0283D", "badge": "🔴"}


def construire_vecteur_features(acwr: float, charge_chronique: float, fc_repos: float,
                                 sommeil_h: float, age: int, imc: float,
                                 historique_blessures: int, deficit_calorique_pct: float,
                                 lactate: float, intensite_moyenne_7j: float) -> np.ndarray:
    """Construit le vecteur de features dans l'ordre attendu par les modèles.

    dtype=object en repli : si une valeur reçue est d'un type exotique (liste,
    dict...) plutôt qu'un scalaire, np.array() avec dtype implicite lève une
    ValueError ("inhomogeneous shape") avant même d'atteindre la validation.
    valider_et_nettoyer_features() sait déjà gérer un vecteur object (elle
    convertit chaque valeur individuellement via float()), donc ce repli est
    sûr : la valeur exotique sera simplement traitée comme manquante.
    """
    valeurs = [acwr, charge_chronique, fc_repos, sommeil_h, age, imc,
               historique_blessures, deficit_calorique_pct, lactate, intensite_moyenne_7j]
    try:
        return np.array([valeurs], dtype=float)
    except (TypeError, ValueError):
        return np.array([valeurs], dtype=object)


def valider_et_nettoyer_features(vecteur) -> tuple:
    """
    Valide et nettoie un vecteur de features avant inférence.
    Retourne (vecteur_nettoye, rapport). Ne lève jamais d'exception.
    """
    rapport = {"champs_imputes": [], "champs_clippes": []}
    n_features = len(FEATURE_NAMES)
    vecteur_nettoye = np.full((1, n_features), np.nan, dtype=float)

    valeurs_brutes = []
    try:
        arr = np.asarray(vecteur, dtype=object)
        valeurs_brutes = list(arr.reshape(-1))
    except Exception:
        valeurs_brutes = []

    while len(valeurs_brutes) < n_features:
        valeurs_brutes.append(np.nan)

    for i, nom in enumerate(FEATURE_NAMES):
        brut = valeurs_brutes[i]
        try:
            val = float(brut)
        except (TypeError, ValueError):
            val = np.nan

        if nom not in DOMAINE_ENTRAINEMENT:
            vecteur_nettoye[0, i] = val if np.isfinite(val) else 0.0
            continue

        borne_min, borne_max = DOMAINE_ENTRAINEMENT[nom]

        if not np.isfinite(val):
            # Valeur typique de l'entraînement (pas le milieu du domaine : voir VALEURS_TYPIQUES).
            vecteur_nettoye[0, i] = VALEURS_TYPIQUES[nom]
            rapport["champs_imputes"].append({"champ": nom, "valeur_imputee": round(float(VALEURS_TYPIQUES[nom]), 3)})
        elif val < borne_min:
            vecteur_nettoye[0, i] = borne_min
            rapport["champs_clippes"].append({"champ": nom, "valeur_recue": round(float(val), 3),
                                               "valeur_utilisee": round(float(borne_min), 3)})
        elif val > borne_max:
            vecteur_nettoye[0, i] = borne_max
            rapport["champs_clippes"].append({"champ": nom, "valeur_recue": round(float(val), 3),
                                               "valeur_utilisee": round(float(borne_max), 3)})
        else:
            vecteur_nettoye[0, i] = val

    return vecteur_nettoye, rapport


def predire_profil_risque(features: np.ndarray) -> dict:
    """Prédit le profil de risque complet d'un athlète à partir de son vecteur de features."""
    clf, reg, metadata = _charger_modeles()

    features_nettoyees, rapport_qualite = valider_et_nettoyer_features(features)

    proba_blessure = float(clf.predict_proba(features_nettoyees)[0, 1])
    injury_risk_pct = round(proba_blessure * 100, 1)

    fatigue_score = float(np.clip(reg.predict(features_nettoyees)[0], 0, 10))

    acwr_val = float(features_nettoyees[0][0])
    overtraining_score = float(np.clip(
        0.45 * (fatigue_score / 10 * 100) +
        0.35 * injury_risk_pct +
        0.20 * max(0, (acwr_val - 1.3)) * 150,
        0, 100
    ))

    importances = metadata.get("feature_importances", [1 / len(FEATURE_NAMES)] * len(FEATURE_NAMES))
    valeurs = features_nettoyees[0]
    contributions = sorted(
        zip(FEATURE_NAMES, importances, valeurs), key=lambda x: x[1], reverse=True
    )[:3]
    facteurs_cles = [
        {"facteur": LABELS_LISIBLES.get(nom, nom), "poids_relatif": round(float(imp), 3),
         "valeur_observee": round(float(val), 2)}
        for nom, imp, val in contributions
    ]

    return {
        "injury_risk_pct": injury_risk_pct,
        "injury_risk_categorie": _categorie_risque(injury_risk_pct),
        "fatigue_score": round(fatigue_score, 1),
        "fatigue_categorie": _categorie_risque(fatigue_score * 10),
        "overtraining_pct": round(overtraining_score, 1),
        "overtraining_categorie": _categorie_risque(overtraining_score),
        "facteurs_cles": facteurs_cles,
        "metadata_modele": metadata,
        "qualite_donnees": rapport_qualite,
        "probabilite_prevalence_pct": round(corriger_prevalence(proba_blessure) * 100, 1),
    }


def predire_tendance_risque(features_base: np.ndarray, df_charge, jours: int = 7) -> tuple[list[dict], dict]:
    """Retourne (trajectoire, sens) : trajectoire de l'Injury Risk Score sur les
    derniers `jours` et résumé de la tendance (hausse / baisse / stable)."""
    clf, _, _ = _charger_modeles()
    features_base_nettoye, _ = valider_et_nettoyer_features(features_base)
    # Jours calendaires (séances du même jour additionnées, repos = 0), comme calculer_acwr.
    charges = charge_quotidienne(df_charge, jours=28 + jours).reset_index(drop=True)
    n = len(charges)
    trajectoire = []

    for i in range(jours - 1, -1, -1):
        fin = n - i
        if fin <= 0:
            continue
        fenetre_aigue = charges.iloc[max(0, fin - 7):fin]
        fenetre_chronique = charges.iloc[max(0, fin - 28):fin]
        chronique = fenetre_chronique.mean() if len(fenetre_chronique) else 0
        acwr_jour = (fenetre_aigue.mean() / chronique) if chronique > 0 else 0

        vec = features_base_nettoye.copy()
        vec[0][0] = acwr_jour
        vec, _ = valider_et_nettoyer_features(vec)
        proba = float(clf.predict_proba(vec)[0, 1])
        trajectoire.append({"jour_offset": -i, "acwr": round(float(acwr_jour), 2),
                             "injury_risk_pct": round(proba * 100, 1)})

    if len(trajectoire) >= 2:
        delta = trajectoire[-1]["injury_risk_pct"] - trajectoire[0]["injury_risk_pct"]
        if delta > 3:
            sens = {"direction": "hausse", "icone": "📈", "texte": f"+{delta:.1f} pts sur {len(trajectoire)}j"}
        elif delta < -3:
            sens = {"direction": "baisse", "icone": "📉", "texte": f"{delta:.1f} pts sur {len(trajectoire)}j"}
        else:
            sens = {"direction": "stable", "icone": "➡️", "texte": "stable"}
    else:
        sens = {"direction": "stable", "icone": "➡️", "texte": "stable"}

    return trajectoire, sens
