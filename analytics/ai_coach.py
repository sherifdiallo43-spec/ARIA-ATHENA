# ATHÉNA - analytics/ai_coach.py
# Coach virtuel : traduit les sorties des modèles ML (Injury Risk, Fatigue,
# Overtraining) et des métriques métier (ACWR, nutrition) en recommandations
# concrètes, priorisées et actionnables pour l'athlète et le staff technique.


def generer_recommandations_coach(profil_risque: dict, acwr_info: dict,
                                   bilan_calorique: dict, sommeil_h: float) -> list:
    """
    Génère une liste de recommandations priorisées (priorité, catégorie, texte)
    à partir du profil de risque ML et des métriques d'entraînement/nutrition.
    """
    reco = []

    injury_pct = profil_risque["injury_risk_pct"]
    fatigue = profil_risque["fatigue_score"]
    overtraining = profil_risque["overtraining_pct"]

    # --- Charge d'entraînement ---
    if injury_pct >= 66 or overtraining >= 66:
        reduction = 25 if injury_pct >= 80 else 15
        reco.append({
            "priorite": 1, "categorie": "Charge",
            "texte": f"Réduire l'intensité d'entraînement de {reduction}% pendant 48h et repasser en zone ACWR optimale (0.8–1.3).",
        })
    elif acwr_info["zone"] == "Prudence":
        reco.append({
            "priorite": 2, "categorie": "Charge",
            "texte": "Stabiliser la charge cette semaine, éviter toute augmentation brutale du volume.",
        })
    elif acwr_info["zone"] == "Sous-charge":
        reco.append({
            "priorite": 3, "categorie": "Charge",
            "texte": "Charge en dessous du seuil optimal : augmenter progressivement le volume (+10%/semaine max).",
        })
    else:
        reco.append({
            "priorite": 4, "categorie": "Charge",
            "texte": "Charge d'entraînement dans la zone optimale : maintenir la progression actuelle.",
        })

    # --- Récupération / sommeil ---
    if sommeil_h < 6.5:
        reco.append({
            "priorite": 1, "categorie": "Récupération",
            "texte": f"Dette de sommeil détectée ({sommeil_h}h) : viser 8h minimum cette nuit, envisager une sieste de 20-30 min.",
        })
    if fatigue >= 6.5:
        reco.append({
            "priorite": 1, "categorie": "Récupération",
            "texte": "Fatigue élevée : prévoir une séance de récupération active (mobilité, natation légère, étirements).",
        })
    elif fatigue >= 4:
        reco.append({
            "priorite": 3, "categorie": "Récupération",
            "texte": "Fatigue modérée : intégrer 10-15 min d'étirements/mobilité en fin de séance.",
        })

    # --- Nutrition ---
    if bilan_calorique["tdee"] > 0:
        ecart = (bilan_calorique["kcal_consommees"] - bilan_calorique["tdee"]) / bilan_calorique["tdee"]
        if ecart < -0.15:
            reco.append({
                "priorite": 2, "categorie": "Nutrition",
                "texte": "Déficit calorique marqué : augmenter les apports (glucides complexes) pour soutenir la récupération.",
            })
        elif ecart > 0.20:
            reco.append({
                "priorite": 3, "categorie": "Nutrition",
                "texte": "Surplus calorique : ajuster les portions, privilégier les protéines maigres.",
            })
    reco.append({
        "priorite": 4, "categorie": "Nutrition",
        "texte": "Augmenter l'hydratation à 35-40 mL/kg/jour, davantage si séance à forte intensité.",
    })

    # --- Blessure ---
    if injury_pct >= 66:
        reco.append({
            "priorite": 1, "categorie": "Prévention blessure",
            "texte": "Risque de blessure élevé : signaler au staff médical avant la prochaine séance à haute intensité.",
        })
    elif injury_pct >= 33:
        reco.append({
            "priorite": 2, "categorie": "Prévention blessure",
            "texte": "Renforcer le travail préventif (proprioception, gainage, renforcement excentrique).",
        })

    reco.sort(key=lambda r: r["priorite"])
    return reco
