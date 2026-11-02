# ATHÉNA - analytics/fiche_athlete.py
# Fiche d'UN athlète venu d'un CSV (onglet Test CSV) : mêmes fonctions et même structure que la démo
# guidée (analytics/scenario_demo.py) : alerte -> facteurs -> plan d'action -> coach -> rapport PDF.
# Rien n'est inventé : ce qu'un CSV ne contient pas (constantes vitales, régularité) n'est pas affiché.

import os
import tempfile

import numpy as np

from analytics.ai_coach import generer_recommandations_coach
from analytics.metrics import alertes_charge_nutrition, calculer_performance_score, zone_acwr
from models.explicabilite import expliquer_prediction, recommander_leviers
from models.injury_risk import (_charger_modeles, corriger_prevalence, predire_profil_risque,
                                valider_et_nettoyer_features)
from models.testeur_csv import RESULTATS, SEUIL_ALERTE
from models.train_injury_model import FEATURE_NAMES


def _alertes(profil: dict, acwr_info: dict, bilan: dict) -> list:
    niveau, score = profil["injury_risk_categorie"]["niveau"], profil["injury_risk_pct"]
    alertes = []
    if niveau == "Élevé":
        alertes.append(("Critique", f"Score de risque élevé ({score} / 100)",
                        "Signaler au staff médical avant la prochaine séance à haute intensité."))
    elif niveau == "Modéré":
        alertes.append(("Attention", f"Score de risque modéré ({score} / 100)",
                        "Renforcer le travail préventif et surveiller la charge."))
    alertes += alertes_charge_nutrition(acwr_info, bilan)
    if not alertes:
        alertes.append(("OK", "Aucune anomalie détectée", "Profil dans les paramètres normaux."))
    priorite = {"Critique": 0, "Attention": 1, "Info": 2, "OK": 3}
    return sorted(alertes, key=lambda a: priorite[a[0]])


def construire_fiche(vecteur_brut, nom: str, blessure_reelle: bool | None = None) -> dict:
    """`vecteur_brut` : les 10 variables converties dans les unités du modèle, NaN si absentes
    (ligne de `res["_X"]` de models.testeur_csv.analyser). Le nettoyage est celui de la production."""
    brut = np.asarray(vecteur_brut, dtype=float).reshape(1, -1)
    propre, qualite = valider_et_nettoyer_features(brut)
    valeurs = {f: float(propre[0, j]) for j, f in enumerate(FEATURE_NAMES)}
    profil = predire_profil_risque(brut)
    clf, _, _ = _charger_modeles()
    score = float(clf.predict_proba(propre)[0, 1])

    acwr_info = {"acwr": round(valeurs["acwr"], 2), **zone_acwr(valeurs["acwr"])}
    # Le modèle travaille en écart relatif au besoin énergétique : un besoin de 1 suffit pour
    # réutiliser le coach et les alertes (écart = -déficit). Sans mesure dans le fichier : non renseigné.
    deficit_mesure = bool(np.isfinite(brut[0, FEATURE_NAMES.index("deficit_calorique_pct")]))
    bilan = {"tdee": 1.0, "kcal_consommees": 1 - valeurs["deficit_calorique_pct"] / 100, "renseigne": deficit_mesure}

    plan = recommander_leviers(propre)
    pc = plan["plan_combine"]
    pc["proba_avant_pct"] = profil["probabilite_prevalence_pct"]   # même valeur que l'alerte (démo : idem)
    pc["proba_apres_pct"] = round(corriger_prevalence(pc["apres_pct"] / 100) * 100, 1)

    verite = None
    if blessure_reelle is not None:
        predit = score >= SEUIL_ALERTE
        verite = {"blessure_reelle": bool(blessure_reelle),
                  "resultat": RESULTATS[(int(predit), int(bool(blessure_reelle)))]}

    return {
        "athlete": {"nom": nom}, "features": propre, "valeurs": valeurs, "qualite": qualite,
        "score_brut": score, "blessure_predite": score >= SEUIL_ALERTE,
        "profil_risque": profil, "acwr_info": acwr_info, "bilan": bilan,
        "explication": expliquer_prediction(propre), "plan_action": plan,
        "recommandations": generer_recommandations_coach(profil, acwr_info, bilan, round(valeurs["sommeil_h"], 1)),
        "performance": calculer_performance_score(acwr_info, profil["injury_risk_pct"], profil["fatigue_score"],
                                                  bilan, None),
        "alertes": _alertes(profil, acwr_info, bilan),
        "verite": verite,
    }


def generer_pdf_fiche(fiche: dict) -> bytes:
    from reports.pdf_report import generer_rapport_pdf
    descripteur, chemin = tempfile.mkstemp(prefix="athena_fiche_", suffix=".pdf")
    os.close(descripteur)
    try:
        generer_rapport_pdf(chemin, fiche["athlete"], fiche["profil_risque"], fiche["acwr_info"],
                            fiche["performance"], fiche["recommandations"], fiche["alertes"],
                            explication=fiche["explication"], plan_action=fiche["plan_action"])
        with open(chemin, "rb") as f:
            return f.read()
    finally:
        os.remove(chemin)
