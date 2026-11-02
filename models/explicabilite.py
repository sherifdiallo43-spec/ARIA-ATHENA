# ATHÉNA - models/explicabilite.py
# Explicabilité INDIVIDUELLE et simulateur "What-If" du risque de blessure.
#
# 1) expliquer_prediction : pour CET athlète, de combien chaque facteur
#    augmente/diminue son risque par rapport à une valeur de référence saine
#    (méthode par substitution, façon "occlusion"). Contrairement à
#    l'importance globale du modèle, ce résultat dépend de l'athlète.
# 2) recommander_leviers : quels changements ACTIONNABLES (sommeil, charge,
#    intensité, apports) réduisent le plus son risque, et de combien.
# 3) simuler_scenario : recalcule le risque si on modifie des valeurs.
#
# Tout passe par le modèle de production : aucune règle inventée à côté.

import numpy as np

from .train_injury_model import FEATURE_NAMES
from .injury_risk import (
    _charger_modeles, valider_et_nettoyer_features, LABELS_LISIBLES,
)

# Valeurs de référence "athlète en bonne santé" (point de comparaison).
REFERENCE_SAINE = {
    "acwr": 1.0, "charge_chronique": 420.0, "fc_repos_estimee": 58.0,
    "sommeil_h": 8.0, "imc": 22.0, "historique_blessures": 0.0,
    "deficit_calorique_pct": 0.0, "lactate": 1.5, "intensite_moyenne_7j": 0.5,
}  # "age" volontairement absent : on ne peut pas "rajeunir" un athlète.

# Leviers sur lesquels le staff peut agir -> (valeur cible, phrase d'action).
LEVIERS = {
    "sommeil_h": (lambda v: max(v, 8.0), "Dormir 8 h par nuit"),
    "acwr": (lambda v: min(v, 1.0) if v > 1.0 else max(v, 0.9), "Ramener l'ACWR vers 1.0"),
    "intensite_moyenne_7j": (lambda v: max(0.05, v - 0.15), "Baisser l'intensité moyenne de 15 points"),
    "deficit_calorique_pct": (lambda v: 0.0, "Couvrir le besoin énergétique (TDEE)"),
}


def _risque_pct(vecteur: np.ndarray) -> float:
    clf, _, _ = _charger_modeles()
    return float(clf.predict_proba(vecteur)[0, 1]) * 100


def _vecteur(valeurs: dict) -> np.ndarray:
    v, _ = valider_et_nettoyer_features(
        np.array([[valeurs[n] for n in FEATURE_NAMES]], dtype=float)
    )
    return v


def _depuis_features(features) -> dict:
    v, _ = valider_et_nettoyer_features(features)
    return {n: float(v[0, i]) for i, n in enumerate(FEATURE_NAMES)}


def expliquer_prediction(features, top: int = 5) -> list[dict]:
    """Contribution de chaque facteur au risque de CET athlète (en points de %).

    Positif = ce facteur augmente son risque vs la référence saine.
    """
    base = _depuis_features(features)
    risque = _risque_pct(_vecteur(base))
    sorties = []
    for nom in FEATURE_NAMES:
        if nom not in REFERENCE_SAINE:
            continue
        alt = dict(base)
        alt[nom] = REFERENCE_SAINE[nom]
        delta = risque - _risque_pct(_vecteur(alt))
        sorties.append({
            "facteur": LABELS_LISIBLES.get(nom, nom), "cle": nom,
            "valeur": round(base[nom], 2), "reference": REFERENCE_SAINE[nom],
            "contribution_pts": round(delta, 1),
        })
    sorties.sort(key=lambda x: abs(x["contribution_pts"]), reverse=True)
    return sorties[:top]


def simuler_scenario(features, modifications: dict) -> dict:
    """Risque avant / après si on applique `modifications` ({feature: valeur})."""
    base = _depuis_features(features)
    avant = _risque_pct(_vecteur(base))
    alt = dict(base)
    alt.update({k: float(v) for k, v in modifications.items() if k in alt})
    apres = _risque_pct(_vecteur(alt))
    return {"avant_pct": round(avant, 1), "apres_pct": round(apres, 1),
            "gain_pts": round(avant - apres, 1)}


def recommander_leviers(features) -> dict:
    """Leviers actionnables classés par gain de risque, + plan combiné."""
    base = _depuis_features(features)
    avant = _risque_pct(_vecteur(base))
    gains, combine = [], {}
    for nom, (cible, phrase) in LEVIERS.items():
        nouvelle = cible(base[nom])
        if abs(nouvelle - base[nom]) < 1e-6:
            continue
        alt = dict(base)
        alt[nom] = nouvelle
        gain = avant - _risque_pct(_vecteur(alt))
        if gain > 0.2:  # on ignore les effets négligeables
            gains.append({"action": phrase, "cle": nom, "de": round(base[nom], 2),
                          "vers": round(nouvelle, 2), "gain_pts": round(gain, 1)})
            combine[nom] = nouvelle
    gains.sort(key=lambda g: g["gain_pts"], reverse=True)
    plan = simuler_scenario(features, combine) if combine else {
        "avant_pct": round(avant, 1), "apres_pct": round(avant, 1), "gain_pts": 0.0}
    return {"leviers": gains, "plan_combine": plan}


def analyser_lot(Xn: np.ndarray) -> dict:
    """Version groupée d'expliquer_prediction et de recommander_leviers pour tout un fichier.

    `Xn` est déjà nettoyé (models.testeur_csv.nettoyer_matrice). Mêmes substitutions que la version
    individuelle, mais une seule prédiction par facteur pour toutes les lignes : rapide sur 3 000 athlètes.
    Retourne le score (pts), la contribution de chaque facteur et le gain de chaque levier (0 si négligeable).
    """
    clf, _, _ = _charger_modeles()
    risque = clf.predict_proba(Xn)[:, 1] * 100
    contributions = {}
    for nom, ref in REFERENCE_SAINE.items():
        alt = Xn.copy()
        alt[:, FEATURE_NAMES.index(nom)] = ref
        contributions[nom] = risque - clf.predict_proba(alt)[:, 1] * 100
    gains = {}
    for nom, (cible, _) in LEVIERS.items():
        j = FEATURE_NAMES.index(nom)
        nouvelles = np.array([cible(v) for v in Xn[:, j]], dtype=float)
        alt = Xn.copy()
        alt[:, j] = nouvelles
        gain = risque - clf.predict_proba(alt)[:, 1] * 100
        gains[nom] = np.where((np.abs(nouvelles - Xn[:, j]) >= 1e-6) & (gain > 0.2), gain, 0.0)
    return {"risque_pct": risque, "contributions": contributions, "gains_leviers": gains}
