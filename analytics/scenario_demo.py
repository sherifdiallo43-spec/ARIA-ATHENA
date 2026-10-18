# ATHÉNA - analytics/scenario_demo.py
# Scénario de démonstration GUIDÉ, 100 % déterministe et hors-ligne : une athlète FICTIVE en
# surcharge d'entraînement, qui traverse toute la chaîne ATHÉNA (alerte -> facteurs -> what-if ->
# plan d'action -> rapport PDF) avec les mêmes fonctions que le dashboard réel.
#
# Les entrées sont fixées à la main pour que la démo se déroule de la même façon à chaque fois ;
# les SORTIES (risque, facteurs, gains) sont calculées par les vrais modèles, pas écrites en dur.
# Ce scénario est fictif et ne représente aucune personne réelle.

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.ai_coach import generer_recommandations_coach
from analytics.metrics import (calculer_acwr, calculer_performance_score, moteur_de_decision)
from data.simulators import generer_micronutriments_simules
from models.explicabilite import expliquer_prediction, recommander_leviers, simuler_scenario
from models.injury_risk import construire_vecteur_features, predire_profil_risque

ATHLETE_DEMO = {"nom": "Awa D. (scénario fictif)", "sexe": "Femme", "age": 21, "poids": 58.0,
                "taille": 168.0, "intensite_effort": 0.78}

# 3 semaines de charge modérée puis 1 semaine de montée brutale (pic de charge typique avant une compétition)
_CHARGE_JOURS = [300, 320, 0, 310, 330, 280, 0] * 3 + [520, 560, 600, 0, 580, 610, 540]

VITALS_DEMO = {"fc": 172, "spo2": 96, "temperature": 37.4, "lactate": 6.8, "glucose": 85}
SOMMEIL_H = 5.4
FC_REPOS = 66
HISTORIQUE_BLESSURES = 2


def construire_scenario() -> dict:
    df_charge = pd.DataFrame({"charge": _CHARGE_JOURS, "jour": range(len(_CHARGE_JOURS))})
    acwr_info = calculer_acwr(df_charge)
    imc = ATHLETE_DEMO["poids"] / (ATHLETE_DEMO["taille"] / 100) ** 2
    bilan = {"tdee": 2500.0, "kcal_consommees": 2050.0, "renseigne": True}
    deficit_pct = (bilan["kcal_consommees"] - bilan["tdee"]) / bilan["tdee"] * 100

    features = construire_vecteur_features(
        acwr=acwr_info["acwr"], charge_chronique=acwr_info["chronique"], fc_repos=FC_REPOS,
        sommeil_h=SOMMEIL_H, age=ATHLETE_DEMO["age"], imc=imc, historique_blessures=HISTORIQUE_BLESSURES,
        deficit_calorique_pct=deficit_pct, lactate=VITALS_DEMO["lactate"],
        intensite_moyenne_7j=ATHLETE_DEMO["intensite_effort"])
    profil_risque = predire_profil_risque(features)
    df_micro = generer_micronutriments_simules(graine=1)
    performance = calculer_performance_score(acwr_info, profil_risque["injury_risk_pct"],
                                             profil_risque["fatigue_score"], bilan, 70.0)
    return {
        "athlete": ATHLETE_DEMO, "df_charge": df_charge, "acwr_info": acwr_info, "features": features,
        "profil_risque": profil_risque, "bilan": bilan, "performance": performance,
        "explication": expliquer_prediction(features), "plan_action": recommander_leviers(features),
        "recommandations": generer_recommandations_coach(profil_risque, acwr_info, bilan, SOMMEIL_H),
        "alertes": moteur_de_decision(VITALS_DEMO, acwr_info, bilan, df_micro),
        "what_if_sommeil": simuler_scenario(features, {"sommeil_h": SOMMEIL_H + 1.5}),
        "what_if_charge": simuler_scenario(features, {"acwr": 1.1}),
    }


def generer_pdf_demo(scenario: dict, chemin: str | None = None, etude: dict | None = None) -> str:
    from reports.pdf_report import generer_rapport_pdf
    chemin = chemin or str(Path(tempfile.gettempdir()) / "athena_demo_rapport.pdf")
    return generer_rapport_pdf(chemin, scenario["athlete"], scenario["profil_risque"], scenario["acwr_info"],
                               scenario["performance"], scenario["recommandations"], scenario["alertes"],
                               etude_scientifique=etude, explication=scenario["explication"],
                               plan_action=scenario["plan_action"])
