# ATHÉNA - models/injury_risk.py
# Wrapper d'inférence pour les modèles ML entraînés (Gradient Boosting).
# Fournit : Injury Risk Score (%), Fatigue Level, Overtraining Risk,
# ainsi que les facteurs contributifs principaux (feature importance).

import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import joblib

from .train_injury_model import FEATURE_NAMES, ARTIFACTS_DIR, entrainer_et_sauvegarder

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


@lru_cache(maxsize=1)
def _charger_modeles():
    """Charge les modèles depuis le disque ; les entraîne si absents (1er lancement)."""
    chemin_clf = ARTIFACTS_DIR / "injury_risk_model.joblib"
    chemin_reg = ARTIFACTS_DIR / "fatigue_model.joblib"
    chemin_meta = ARTIFACTS_DIR / "metadata.json"

    if not (chemin_clf.exists() and chemin_reg.exists()):
        entrainer_et_sauvegarder()

    try:
        clf = joblib.load(chemin_clf)
        reg = joblib.load(chemin_reg)
    except Exception:
        # Fichiers .joblib incompatibles avec la version de scikit-learn/numpy
        # installée localement (ex. changement de version) -> on réentraîne
        # à la volée plutôt que de planter le dashboard.
        entrainer_et_sauvegarder()
        clf = joblib.load(chemin_clf)
        reg = joblib.load(chemin_reg)

    metadata = json.loads(chemin_meta.read_text()) if chemin_meta.exists() else {}
    return clf, reg, metadata


def _categorie_risque(pct: float) -> dict:
    if pct < 33:
        return {"niveau": "Faible", "couleur": "#06D6A0", "badge": "🟢"}
    elif pct < 66:
        return {"niveau": "Modéré", "couleur": "#FFB703", "badge": "🟠"}
    else:
        return {"niveau": "Élevé", "couleur": "#FF2E63", "badge": "🔴"}


def construire_vecteur_features(acwr: float, charge_chronique: float, fc_repos: float,
                                 sommeil_h: float, age: int, imc: float,
                                 historique_blessures: int, deficit_calorique_pct: float,
                                 lactate: float, intensite_moyenne_7j: float) -> np.ndarray:
    """Construit le vecteur de features dans l'ordre attendu par les modèles."""
    return np.array([[
        acwr, charge_chronique, fc_repos, sommeil_h, age, imc,
        historique_blessures, deficit_calorique_pct, lactate, intensite_moyenne_7j,
    ]])


def predire_profil_risque(features: np.ndarray) -> dict:
    """
    Prédit le profil de risque complet d'un athlète à partir de son vecteur
    de features. Retourne Injury Risk Score, Fatigue Level, Overtraining Risk,
    et les 3 facteurs contributifs les plus importants (explicabilité).
    """
    clf, reg, metadata = _charger_modeles()

    proba_blessure = float(clf.predict_proba(features)[0, 1])
    injury_risk_pct = round(proba_blessure * 100, 1)

    fatigue_score = float(np.clip(reg.predict(features)[0], 0, 10))

    acwr_val = float(features[0][0])
    overtraining_score = float(np.clip(
        0.45 * (fatigue_score / 10 * 100) +
        0.35 * injury_risk_pct +
        0.20 * max(0, (acwr_val - 1.3)) * 150,
        0, 100
    ))

    # Facteurs contributifs (feature importance du modèle de classification,
    # stockée dans metadata.json car clf est un CalibratedClassifierCV et
    # n'expose pas .feature_importances_ directement)
    importances = metadata.get("feature_importances", [1 / len(FEATURE_NAMES)] * len(FEATURE_NAMES))
    valeurs = features[0]
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
    }


def predire_tendance_risque(features_base: np.ndarray, df_charge, jours: int = 7) -> list[dict]:
    """
    Trajectoire de l'Injury Risk Score sur les derniers `jours` : recalcule l'ACWR
    jour par jour (fenêtres glissantes 7j/28j sur l'historique de charge réel) et
    ré-injecte chaque valeur dans le modèle en gardant les autres features fixes.
    Permet d'afficher une tendance ("risque en hausse depuis 3 jours") plutôt
    qu'un score isolé — logique utilisée par les outils de charge d'entraînement
    du type Catapult / Whoop / Zone7.
    """
    clf, _, _ = _charger_modeles()
    charges = df_charge["charge"].reset_index(drop=True)
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

        vec = features_base.copy()
        vec[0][0] = acwr_jour
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
