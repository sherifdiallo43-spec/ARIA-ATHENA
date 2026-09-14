# ATHÉNA v2 — Plateforme IA Sportive Prédictive
ARiA Tech — Pan-African Robotics Competition 2026, Ligue Intelligence Artificielle

## Fonctionnalités

| Fonctionnalité | Implémentation |
|---|---|
| Prédiction du risque de blessure | `models/injury_risk.py` — Gradient Boosting (scikit-learn), Injury Risk Score %, code couleur vert/orange/rouge |
| AI Coach | `analytics/ai_coach.py` — recommandations priorisées (charge, récupération, nutrition, prévention) |
| Analyse vidéo | `computer_vision/pose_analysis.py` — MediaPipe Pose, angles genou/hanche/tronc calibrés par type de mouvement, asymétrie gauche/droite |
| Analyse de performance | `analytics/metrics.py::calculer_performance_score` — Athlete Performance Score /100 + radar chart |
| Rapport IA | `reports/pdf_report.py` — bouton "Generate AI Report" → PDF exportable |
| Mode National Team | Onglet "National Team" — suivi simultané de 6 athlètes (classement, risque, disponibilité) |

## Architecture

```
athena/
├── data/            # Simulateurs physiologiques (ECG, vitals, charge, micronutriments, équipe)
├── models/          # ML : entraînement + inférence (Injury Risk, Fatigue)
│   └── artifacts/   # Modèles .joblib + métadonnées
├── analytics/        # ACWR, BMR/TDEE, moteur de règles, Performance Score, AI Coach
├── computer_vision/  # Analyse posturale MediaPipe (Pose Landmarker)
│   └── artifacts/    # Modèle .task (voir setup ci-dessous)
├── reports/           # Génération de rapport PDF (ReportLab)
└── dashboard/
    └── app.py         # Point d'entrée Streamlit — assemble tous les modules
```

## Installation & lancement

```bash
cd athena
python3 -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt

# Une seule fois, avec internet : télécharge le modèle de détection posturale
# en local pour que le dashboard n'ait plus besoin de réseau ensuite (utile
# le jour de la démo si le wifi de la salle n'est pas fiable).
python -m computer_vision.pose_analysis --setup

streamlit run dashboard/app.py
```

Les modèles Injury Risk / Fatigue sont déjà entraînés et livrés dans `models/artifacts/`
(joblib). Pour les ré-entraîner (recherche d'hyperparamètres par validation croisée,
~2-3 min) : `python -m models.train_injury_model`.

## Méthodologie IA

- Les modèles Injury Risk / Fatigue sont entraînés sur un jeu de données synthétique
  (8000 échantillons), construit à partir de relations non-linéaires issues de la
  littérature sportive (ACWR — Gabbett et al., dette de sommeil, historique de blessures,
  charge chronique) plutôt que sur du bruit aléatoire.
- Les hyperparamètres (profondeur, nombre d'arbres, learning rate, sous-échantillonnage)
  sont choisis par `RandomizedSearchCV` (5-fold, scoring ROC-AUC / MAE) plutôt que fixés à
  la main — voir `models/artifacts/metadata.json` pour les valeurs retenues.
- Le modèle de classification atteint un AUC ≈ 0.66 sur test, et l'étude comparative
  (onglet "Étude Scientifique") montre que ce chiffre ne bouge presque pas entre
  régression logistique, Random Forest et Gradient Boosting : la limite actuelle vient
  du volume et de la richesse du dataset synthétique, pas du choix de modèle. Le vrai
  levier pour la V3 est le passage à des données de capteurs réels.
- L'architecture est conçue pour être reconnectée à de vraies données de capteurs sans
  changer la logique métier : il suffit de remplacer les simulateurs de
  `data/simulators.py` par les flux capteurs réels.
- Le module Computer Vision est un démonstrateur mono-image (pas un système biomécanique
  certifié médicalement), calibré sur trois types de mouvement (course, réception de
  saut, squat) avec une comparaison gauche/droite pour détecter les asymétries.

## Étude scientifique comparative (Research Question / Hypothesis / Baseline / Results)

Répond aux critères du règlement PARC 2026 (Research Question, Hypothesis, Experiment,
Baseline, Conditions, Results, Discussion). Module : `models/experiment_comparaison.py`,
reproductible via `python -m models.experiment_comparaison` (régénère aussi les figures).
Visible dans le dashboard (onglet "Étude Scientifique") et en annexe du rapport PDF.

Référence scientifique : Leckey C, van Dyk N, Doherty C, Lawlor A, Delahunt E. « Machine
learning approaches to injury risk prediction in sport: a scoping review with evidence
synthesis ». *Br J Sports Med*. 2025;59(7):491-500. doi:10.1136/bjsports-2024-108576.

Research Question : un modèle non-linéaire à ensemble d'arbres apporte-t-il un gain
mesurable, par rapport à une régression logistique (baseline) et une forêt aléatoire,
pour prédire le risque de blessure et la fatigue ?

Hypothèse :
- H1 — les modèles non-linéaires à ensemble d'arbres (Gradient Boosting, Random Forest)
  prédisent mieux le risque de blessure sportive que les modèles linéaires, mesuré par
  le ROC-AUC.
- H0 — aucune différence significative de ROC-AUC entre les familles de modèles.
- Test : bootstrap apparié sur l'échantillon de test (1000 tirages).

Conditions : même dataset synthétique, même split 80/20 (`random_state=42`), mêmes 10
features, pondération équilibrée (`class_weight`/`sample_weight`) appliquée aux 3 modèles.

Résultats détaillés, validation croisée 5-fold, test d'hypothèse et discussion : voir
l'onglet "Étude Scientifique" du dashboard (régénérés à chaque exécution de
`experiment_comparaison.py`, donc toujours à jour avec le modèle courant).

## Ce qui reste simulé

- Toutes les constantes vitales (FC, ECG, SpO2, température, lactate, glucose) : simulateur
  physiologique réaliste, en attente de capteurs Bluetooth/I2C (points d'injection prévus).
- Les profils de l'équipe nationale (mode "National Team") sont fictifs, pour la
  démonstration.

## Pistes d'amélioration

1. Enregistrer une courte vidéo de démonstration montrant le flux complet : sélection
   athlète → alerte IA → recommandation coach → rapport PDF généré.
2. Ajouter quelques vraies mesures (même prises au smartphone : FC via appli, sommeil
   déclaré) pour montrer une bascule simulation → données réelles, même partielle.
3. Étendre l'analyse vidéo à une séquence de plusieurs images (course/appuis successifs)
   plutôt qu'une image isolée.
