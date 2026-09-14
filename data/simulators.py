# ATHÉNA - data/simulators.py
# Simulateurs de données physiologiques et d'entraînement.
# Ces fonctions remplacent les capteurs physiques absents (ECG, FC, SpO2,
# température, lactate, glucose, charge d'entraînement) par des modèles
# statistiques réalistes, afin de garantir une démonstration fluide.

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generer_battement_ecg_synthetique(n_points: int = 120) -> np.ndarray:
    """Génère un battement cardiaque ECG synthétique (ondes P, Q, R, S, T)."""
    t = np.linspace(-1, 1, n_points)
    onde_p = 0.15 * np.exp(-((t + 0.35) ** 2) / (2 * 0.028 ** 2))
    onde_q = -0.15 * np.exp(-((t + 0.05) ** 2) / (2 * 0.006 ** 2))
    onde_r = 1.20 * np.exp(-((t) ** 2) / (2 * 0.010 ** 2))
    onde_s = -0.25 * np.exp(-((t - 0.05) ** 2) / (2 * 0.008 ** 2))
    onde_t = 0.30 * np.exp(-((t - 0.32) ** 2) / (2 * 0.045 ** 2))
    return onde_p + onde_q + onde_r + onde_s + onde_t


def generer_signal_ecg(hr_bpm: float, duree_s: float = 3.0, fe_hz: int = 250,
                        bruit_std: float = 0.02) -> np.ndarray:
    """Construit un signal ECG continu selon la fréquence cardiaque donnée."""
    n_total = int(duree_s * fe_hz)
    intervalle_rr = int((60.0 / max(hr_bpm, 30)) * fe_hz)
    n_points_battement = max(int(intervalle_rr * 0.6), 40)

    battement = generer_battement_ecg_synthetique(n_points_battement)
    signal = np.zeros(n_total)

    position = 0
    while position < n_total:
        fin = min(position + n_points_battement, n_total)
        signal[position:fin] += battement[: fin - position]
        position += intervalle_rr

    bruit = np.random.normal(0, bruit_std, n_total)
    return signal + bruit


def simuler_constantes_vitales(profil: dict, etat_precedent: dict | None = None) -> dict:
    """Simule une nouvelle mesure des constantes vitales (marche aléatoire bornée)."""
    intensite = profil.get("intensite_effort", 0.5)

    if etat_precedent is None:
        base_hr = 65 + intensite * 90
        etat_precedent = {
            "fc": base_hr, "spo2": 98.0, "temperature": 36.8,
            "lactate": 1.2, "glucose": 95.0,
        }

    fc = etat_precedent["fc"] + np.random.normal(0, 2.5) + (intensite - 0.5) * 3
    fc = float(np.clip(fc, 45, 205))

    spo2 = etat_precedent["spo2"] + np.random.normal(0, 0.3) - intensite * 0.05
    spo2 = float(np.clip(spo2, 88, 100))

    temperature = etat_precedent["temperature"] + np.random.normal(0, 0.04) + intensite * 0.01
    temperature = float(np.clip(temperature, 35.5, 40.0))

    lactate = etat_precedent["lactate"] + np.random.normal(0, 0.15) + intensite * 0.25
    lactate = float(np.clip(lactate, 0.5, 14.0))

    glucose = etat_precedent["glucose"] + np.random.normal(0, 1.2) - intensite * 0.3
    glucose = float(np.clip(glucose, 60, 180))

    return {
        "fc": round(fc, 1), "spo2": round(spo2, 1), "temperature": round(temperature, 2),
        "lactate": round(lactate, 2), "glucose": round(glucose, 1),
        "horodatage": datetime.now(),
    }


def generer_historique_charge(jours: int = 28, graine: int | None = None) -> pd.DataFrame:
    """Simule un historique de charge d'entraînement journalière (Session-RPE)."""
    rng = np.random.default_rng(graine)
    dates = [datetime.now().date() - timedelta(days=i) for i in range(jours - 1, -1, -1)]
    charges = np.clip(rng.normal(420, 130, jours), 80, 900)
    return pd.DataFrame({"date": dates, "charge": charges.round(0)})


def generer_micronutriments_simules(graine: int | None = None) -> pd.DataFrame:
    """Simule un panel de micronutriments avec seuils de carence de référence."""
    rng = random.Random(graine)
    data = {
        "Micronutriment": ["Fer (ferritine)", "Vitamine D", "Magnésium", "Potassium", "Vitamine B12"],
        "Valeur mesurée": [
            round(rng.uniform(15, 120), 1), round(rng.uniform(10, 55), 1),
            round(rng.uniform(1.4, 2.6), 2), round(rng.uniform(3.2, 5.4), 2),
            round(rng.uniform(150, 700), 0),
        ],
        "Seuil minimal": [30, 20, 1.7, 3.5, 200],
        "Unité": ["ng/mL", "ng/mL", "mmol/L", "mmol/L", "pg/mL"],
    }
    return pd.DataFrame(data)


# Profils d'équipe simulés — pour le mode "National Team Dashboard"
EQUIPE_NATIONALE_DEMO = [
    {"nom": "Amara K.", "sport": "Sprint 100m", "age": 21, "poids": 68, "taille": 178,
     "intensite_effort": 0.65, "sommeil_h": 7.2, "historique_blessures": 1},
    {"nom": "Fatou D.", "sport": "Endurance / 5000m", "age": 24, "poids": 54, "taille": 165,
     "intensite_effort": 0.55, "sommeil_h": 6.5, "historique_blessures": 0},
    {"nom": "Kwame O.", "sport": "Football (Milieu)", "age": 26, "poids": 74, "taille": 180,
     "intensite_effort": 0.72, "sommeil_h": 6.8, "historique_blessures": 2},
    {"nom": "Aïcha B.", "sport": "Basketball (Ailière)", "age": 23, "poids": 66, "taille": 182,
     "intensite_effort": 0.60, "sommeil_h": 7.5, "historique_blessures": 0},
    {"nom": "Moussa T.", "sport": "Football (Défenseur)", "age": 28, "poids": 79, "taille": 185,
     "intensite_effort": 0.80, "sommeil_h": 5.9, "historique_blessures": 3},
    {"nom": "Ndeye S.", "sport": "Athlétisme (Saut)", "age": 20, "poids": 58, "taille": 170,
     "intensite_effort": 0.50, "sommeil_h": 8.0, "historique_blessures": 0},
]


def generer_snapshot_equipe(graine: int | None = 42) -> pd.DataFrame:
    """
    Génère un instantané complet (vitals + charge + score) pour toute l'équipe,
    utilisé par le mode National Team Dashboard.
    """
    rng = np.random.default_rng(graine)
    lignes = []
    for athlete in EQUIPE_NATIONALE_DEMO:
        acwr = round(float(np.clip(rng.normal(1.05, 0.28), 0.5, 2.0)), 2)
        fatigue_subjective = round(float(np.clip(rng.normal(5, 2), 1, 10)), 1)
        lignes.append({
            "Nom": athlete["nom"],
            "Sport": athlete["sport"],
            "Âge": athlete["age"],
            "ACWR": acwr,
            "Sommeil (h)": athlete["sommeil_h"],
            "Fatigue subjective (/10)": fatigue_subjective,
            "Historique blessures": athlete["historique_blessures"],
            "Intensité effort": athlete["intensite_effort"],
        })
    return pd.DataFrame(lignes)
