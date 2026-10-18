# ATHÉNA - analytics/metrics.py
# Fonctions métier : ACWR, BMR/TDEE, moteur de règles, Athlete Performance Score

import pandas as pd


def calculer_acwr(df_charge: pd.DataFrame) -> dict:
    """
    Calcule l'Acute:Chronic Workload Ratio (ACWR).
    Zones de risque (littérature sportive - Gabbett et al.) :
        < 0.8   -> sous-entraînement (perte de forme)
        0.8-1.3 -> zone optimale ("sweet spot")
        1.3-1.5 -> zone de prudence
        > 1.5   -> risque élevé de blessure
    """
    if df_charge.empty:
        aigue, chronique = 0.0, 0.0
    else:
        aigue = df_charge.tail(7)["charge"].mean()
        chronique = df_charge.tail(28)["charge"].mean()
        aigue = 0.0 if pd.isna(aigue) else aigue
        chronique = 0.0 if pd.isna(chronique) else chronique
    acwr = aigue / chronique if chronique > 0 else 0

    if acwr < 0.8:
        zone, couleur = "Sous-charge", "#5B9BB0"
    elif acwr <= 1.3:
        zone, couleur = "Zone optimale", "#1F7A52"
    elif acwr <= 1.5:
        zone, couleur = "Prudence", "#B26B00"
    else:
        zone, couleur = "Risque élevé", "#C0283D"

    return {
        "acwr": round(acwr, 2), "aigue": round(aigue, 1), "chronique": round(chronique, 1),
        "zone": zone, "couleur": couleur,
    }


def calculer_bmr_tdee(sexe: str, poids_kg: float, taille_cm: float, age: int,
                       facteur_activite: float) -> dict:
    """Calcule le BMR (Mifflin-St Jeor) puis le TDEE."""
    if sexe == "Homme":
        bmr = 10 * poids_kg + 6.25 * taille_cm - 5 * age + 5
    else:
        bmr = 10 * poids_kg + 6.25 * taille_cm - 5 * age - 161
    tdee = bmr * facteur_activite
    return {"bmr": round(bmr), "tdee": round(tdee)}


def moteur_de_decision(vitals: dict, acwr_info: dict, bilan_calorique: dict,
                        df_micro: pd.DataFrame) -> list:
    """Moteur de règles générant des alertes priorisées pour le staff technique."""
    alertes = []

    if vitals["fc"] > 190:
        alertes.append(("Critique", "Fréquence cardiaque anormalement élevée",
                         "Interrompre l'effort immédiatement et réévaluer l'athlète."))
    if vitals["spo2"] < 94:
        alertes.append(("Critique", "Saturation en oxygène basse (SpO2 < 94%)",
                         "Vérifier la fonction respiratoire, envisager une pause."))
    if vitals["temperature"] > 38.5:
        alertes.append(("Attention", "Température corporelle élevée",
                         "Surveiller un risque d'hyperthermie / coup de chaleur."))
    if vitals["lactate"] > 6:
        alertes.append(("Attention", "Taux de lactate élevé",
                         "Fatigue métabolique importante : réduire l'intensité."))
    if vitals["glucose"] < 70:
        alertes.append(("Attention", "Hypoglycémie détectée",
                         "Apport glucidique rapide recommandé."))

    if acwr_info["zone"] == "Risque élevé":
        alertes.append(("Critique", f"ACWR élevé ({acwr_info['acwr']})",
                         "Risque de blessure accru : réduire la charge cette semaine."))
    elif acwr_info["zone"] == "Sous-charge":
        alertes.append(("Info", f"ACWR faible ({acwr_info['acwr']})",
                         "Perte de forme possible : augmenter progressivement la charge."))
    elif acwr_info["zone"] == "Prudence":
        alertes.append(("Attention", f"ACWR en zone de prudence ({acwr_info['acwr']})",
                         "Surveiller l'évolution de la charge sur les prochains jours."))

    # Si aucun repas n'a été saisi, on ne peut pas conclure à un déficit : la
    # donnée est absente (et non égale à 0 kcal) -> aucune alerte nutritionnelle.
    if bilan_calorique["tdee"] > 0 and bilan_calorique.get("renseigne", True):
        ecart = (bilan_calorique["kcal_consommees"] - bilan_calorique["tdee"]) / bilan_calorique["tdee"]
        if ecart < -0.20:
            alertes.append(("Attention", "Déficit calorique important (> 20%)",
                             "Augmenter les apports pour soutenir la récupération."))
        elif ecart > 0.25:
            alertes.append(("Info", "Surplus calorique important (> 25%)",
                             "Réévaluer les apports par rapport à l'objectif de performance."))

    for _, ligne in df_micro.iterrows():
        if ligne["Valeur mesurée"] < ligne["Seuil minimal"]:
            alertes.append(("Attention", f"Carence détectée : {ligne['Micronutriment']}",
                             "Ajustement diététique ou supplémentation à discuter avec un professionnel."))

    if not alertes:
        alertes.append(("OK", "Aucune anomalie détectée", "L'athlète est dans les paramètres normaux."))

    priorite = {"Critique": 0, "Attention": 1, "Info": 2, "OK": 3}
    alertes.sort(key=lambda a: priorite[a[0]])
    return alertes


def calculer_performance_score(acwr_info: dict, injury_risk_pct: float,
                                fatigue_score: float, bilan_calorique: dict,
                                regularite_pct: float) -> dict:
    """
    Athlete Performance Score (0-100), agrégation pondérée de 5 dimensions :
    condition physique, charge, récupération, nutrition, régularité.
    Retourne le score global + le détail par dimension (pour le radar chart).
    """
    # Condition physique : pénalisée par le risque de blessure ML
    condition_physique = max(0, 100 - injury_risk_pct)

    # Charge : optimale autour d'ACWR = 1.0 (zone Gabbett 0.8-1.3)
    ecart_acwr = abs(acwr_info["acwr"] - 1.0)
    charge = max(0, 100 - ecart_acwr * 90)

    # Récupération : inverse du score de fatigue (0-10 -> 100-0)
    recuperation = max(0, 100 - fatigue_score * 10)

    # Nutrition : proximité du TDEE
    if bilan_calorique["tdee"] > 0 and bilan_calorique.get("renseigne", True):
        ecart_nutrition = abs(bilan_calorique["kcal_consommees"] - bilan_calorique["tdee"]) / bilan_calorique["tdee"]
        nutrition = max(0, 100 - ecart_nutrition * 150)
    else:
        nutrition = 70

    regularite = max(0, min(100, regularite_pct))

    dimensions = {
        "Condition physique": round(condition_physique, 1),
        "Charge d'entraînement": round(charge, 1),
        "Récupération": round(recuperation, 1),
        "Nutrition": round(nutrition, 1),
        "Régularité": round(regularite, 1),
    }

    score_global = round(sum(dimensions.values()) / len(dimensions), 1)

    return {"score_global": score_global, "dimensions": dimensions}
