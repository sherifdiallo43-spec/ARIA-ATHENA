# ATHÉNA v2 — Plateforme IA Sportive Prédictive
ARiA Tech — Pan-African Robotics Competition 2026, Ligue Intelligence Artificielle

## Fonctionnalités

| Fonctionnalité | Implémentation |
|---|---|
| Prédiction du risque de blessure | `models/injury_risk.py` — Gradient Boosting (scikit-learn), Injury Risk Score %, code couleur vert/orange/rouge |
| AI Coach | `analytics/ai_coach.py` — recommandations priorisées (charge, récupération, nutrition, prévention) |
| Analyse vidéo | `computer_vision/pose_analysis.py` — MediaPipe Pose, angles genou/hanche/tronc calibrés par type de mouvement, asymétrie gauche/droite |
| Analyse vidéo sur séquence | `computer_vision/analyse_sequence.py` — vidéo de 5-15 s : angles image par image, amplitude, symétrie gauche/droite, cadence, score évalué à l'image clé de chaque foulée/répétition (voir ci-dessous) |
| Démo guidée | Onglet « Démo Guidée » + `analytics/scenario_demo.py` — scénario fictif déterministe, hors-ligne : alerte → facteurs → what-if → plan d'action → PDF |
| Analyse de performance | `analytics/metrics.py::calculer_performance_score` — Athlete Performance Score /100 + radar chart |
| Rapport IA | `reports/pdf_report.py` — bouton "Generate AI Report" → PDF exportable |
| Mode National Team | Onglet "National Team" — suivi simultané de 6 athlètes (classement, risque, disponibilité) |
| Données réelles | `data/donnees_reelles.py` + onglet "Données Réelles" — journal quotidien (RPE, durée, sommeil, FC au réveil, blessure), features sRPE, score de risque sur athlètes réels, évaluation honnête (IC95 par bootstrap par athlète, baselines ACWR seul / sommeil seul) |

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
# Se placer dans le dossier du projet :
#   archive zip : cd ATHENA        clone GitHub : cd ARIA-ATHENA
cd ATHENA
python3 -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt

# Une seule fois, avec internet : télécharge le modèle de détection posturale
# en local pour que le dashboard n'ait plus besoin de réseau ensuite (utile
# le jour de la démo si le wifi de la salle n'est pas fiable).
# Inutile si computer_vision/artifacts/pose_landmarker_lite.task est déjà présent
# (c'est le cas dans l'archive zip remise au jury).
python -m computer_vision.pose_analysis --setup

streamlit run dashboard/app.py
```

Les modèles Injury Risk / Fatigue sont déjà entraînés et livrés dans `models/artifacts/`
(joblib). Pour les ré-entraîner (recherche d'hyperparamètres par validation croisée,
~2-3 min) : `python -m models.train_injury_model`.

**Lancement en une commande** : `./lancer_demo.sh` (Linux/macOS) ou `lancer_demo.bat` (Windows). La première
fois (avec internet) le script crée l'environnement, installe les dépendances et télécharge le modèle de
pose ; ensuite tout fonctionne sans internet. `python verifier_demo.py` affiche une check-list feu vert /
feu rouge (dépendances, modèles, modèle de pose, tests) : à lancer la veille puis le matin de la démo.

## Dépannage rapide

- **Onglet Analyse Vidéo : « modèle non chargé »** : cliquer sur « Télécharger le modèle » dans l'onglet
  (une seule fois, avec internet) ou lancer `python -m computer_vision.pose_analysis --setup`.
- **Avertissements `use_container_width`** : le dashboard détecte automatiquement l'API Streamlit disponible
  (`width="stretch"` ou ancienne API) — aucune version à figer.
- **Étude scientifique** : les résultats pré-calculés sont livrés (affichage instantané). La régénération
  complète (`python -m models.experiment_comparaison`) prend 2 à 3 minutes.

## Méthodologie IA

- Les modèles Injury Risk / Fatigue sont entraînés sur un jeu de données synthétique
  (8000 échantillons), construit à partir de relations non-linéaires issues de la
  littérature sportive (ACWR — Gabbett et al., dette de sommeil, historique de blessures,
  charge chronique) plutôt que sur du bruit aléatoire.
- Les hyperparamètres (profondeur, nombre d'arbres, learning rate, sous-échantillonnage)
  sont choisis par `RandomizedSearchCV` (5-fold, scoring ROC-AUC / MAE) plutôt que fixés à
  la main — voir `models/artifacts/metadata.json` pour les valeurs retenues.
- Le modèle de classification atteint un AUC de 0.68 ± 0.02 (moyenne ± écart-type sur 10 graines indépendantes ; plage 0.66–0.72), et l'étude comparative
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
features, pondération équilibrée (`class_weight`/`sample_weight`) appliquée aux 4 modèles (Logistic Regression, Random Forest, Gradient Boosting,
HistGradientBoosting).

Robustesse statistique : validation croisée 5-fold + réplication sur 10 graines
aléatoires indépendantes (moyenne ± écart-type, plage min–max par modèle).

Résultats détaillés, validation croisée 5-fold, test d'hypothèse et discussion : voir
l'onglet "Étude Scientifique" du dashboard (régénérés à chaque exécution de
`experiment_comparaison.py`, donc toujours à jour avec le modèle courant).

## Expérience E2 : profondeur des arbres et plafond oracle (pré-enregistrée)
E1 ne montrait pas d'avantage net du Gradient Boosting sur la régression logistique. E2 explique pourquoi en ne faisant
varier qu'un facteur : la profondeur `d` des arbres (ordre d'interaction), avec un **oracle** (AUC du risque latent sans bruit,
optimal en AUC, disponible car les données sont synthétiques). Hypothèses et seuils : `docs/E2_preenregistrement.md`
(écrit avant la première exécution). Résultats : `models/artifacts/experiment_profondeur.json`.
- H2 confirmée : aucune profondeur > 1 ne gagne 0,010 d'AUC ; l'AUC baisse avec la profondeur.
- H3 confirmée : écart à l'oracle 0,007 (seuil 0,030) ; le plafond ≈ 0,70 vient du bruit du générateur.
- H4 confirmée : GB d = 1 bat la régression logistique de +0,020 AUC (IC95 [0,012 ; 0,028]).

```bash
python -m models.experiment_profondeur   # ≈ 2,5 min (2 cœurs) -> experiment_profondeur.json
python -m reports.figures_rapport        # figures du rapport -> reports/rapport_figures/
```
Note : E1 utilise un GB de référence fixe (180 arbres, d = 3), distinct du modèle de production (d = 2, hyperparamètres
issus de RandomizedSearchCV). Les « 10 graines » de E1 sont 10 partitions du même jeu ; E2 (P2) utilise 10 jeux indépendants.

## Expérience E3 : et si les données contiennent une interaction ? (pré-enregistrée)
E2 montre que la profondeur 1 est optimale sur un risque additif. E3 ajoute au risque un terme croisé ACWR × sommeil de
force `c` ∈ {0 ; 0,25 ; 0,5 ; 1,0} (10 jeux indépendants par `c`) pour choisir une profondeur **robuste**. Hypothèses et
seuils : `docs/E3_preenregistrement.md` (écrit avant l'exécution). Résultats : `models/artifacts/experiment_interaction.json`.
- H8 (réplication de E2) confirmée ; H6 confirmée (d = 3 n'apporte rien de plus que d = 2) ; H7 confirmée : regret maximal de
  d = 2 = 0,0096 (seuil 0,012) contre 0,0435 pour d = 1 ; H5 **non concluante** (interaction modérée invisible à n = 6 000).
- Lecture : d = 1 gagne si le risque est additif mais perd 0,044 d'AUC si l'interaction est forte ; d = 2 (profondeur du modèle de
  production) perd au plus 0,010 dans le cas additif et gagne jusqu'à 0,044 sinon : choix minimax.
```bash
python -m models.experiment_interaction   # ≈ 5 min (2 cœurs) -> experiment_interaction.json
```

## Robustesse aux données inédites

`models/injury_risk.py` valide chaque vecteur d'entrée avant inférence
(`valider_et_nettoyer_features`) : valeur manquante / non numérique -> imputée par le
milieu du domaine d'entraînement ; valeur hors domaine -> clippée à la borne la plus
proche ; valeur valide -> jamais modifiée. Le rapport de nettoyage est renvoyé dans la clé
`qualite_donnees`. Test (5 scénarios) : `python -m models.test_robustesse`.

## Analyses avancées (importance, apprentissage, robustesse, calibration)

`python -m models.analyse_avancee` (~1 min) génère `models/artifacts/advanced_analysis.json`
et 3 figures dans `reports/artifacts_figures/` : importance par permutation, courbe
d'apprentissage, courbes de robustesse (bruit, valeurs manquantes). Contenu : importance par
permutation, courbe d'apprentissage, robustesse au bruit / aux manquants, test sur données
inédites (graine 123), fiabilité des scores, analyse des faux négatifs.

Note d'interprétation : le score affiché (`injury_risk_pct`) est un score de risque relatif
(entraînement à classes équilibrées). La probabilité absolue, corrigée de la prévalence
réelle (~20 %), est disponible dans `probabilite_prevalence_pct`.

## Analyse vidéo sur séquence

`computer_vision/analyse_sequence.py` analyse une vidéo courte (5 à 15 s, caméra fixe, corps entier).
Deux couches séparées : l'extraction (MediaPipe, image par image) et l'analyse temporelle, un calcul pur
sur une table d'angles, donc testable sans le modèle de pose.

Choix de méthode (corrigés après tests, voir `models/test_sequence.py`) :
- **Filtre de Hampel** (remplacement des seules valeurs à plus de 25° de la médiane locale) plutôt qu'une
  médiane glissante partout, qui amputait l'amplitude des foulées d'environ 30 % à 10 images/s.
- **Score évalué aux images clés** : extension maximale du genou (appui au sol) en course, flexion
  maximale en squat/réception. Les angles de référence décrivent ces moments précis ; les appliquer à
  chaque image pénaliserait à tort la phase de suspension d'une foulée parfaite.
- **Asymétrie selon le mouvement** : écart instantané gauche/droite pour squat et réception (jambes en
  phase) ; en course les jambes sont en opposition de phase, on compare donc l'amplitude (ROM, percentiles
  5-95) et la flexion maximale de chaque jambe.
- **Refus de conclure** si moins de 50 % des images sont détectées ou si aucun cycle n'est exploitable.

Limites : démonstrateur 2D (une caméra), angles projetés dans le plan de l'image ; pas un outil médical
certifié. Les tests utilisent des séries d'angles fabriquées pour vérifier la logique ; **la détection de
pose réelle sur de vraies vidéos n'a pas pu être testée dans l'environnement de développement** (modèle de
pose non téléchargeable sans internet) : à valider sur 2-3 vidéos filmées par l'équipe avant la démo.

## Évaluation sur jeu de données public

`python -m data.importer_public --fichier <csv> --mapping <json> --nom <nom>` convertit un jeu public vers le
format ATHÉNA (voir `data/mapping_exemple.json`, noms de colonnes fictifs). Le script ne devine aucun format.
L'évaluation exige des données **quotidiennes** (≥ 60 % des jours renseignés) : un jeu hebdomadaire est
refusé explicitement plutôt que d'être évalué de travers. Aucun jeu public n'est fourni avec le projet.

## Calibration et sous-groupes

La courbe de fiabilité (`reports/artifacts_figures/calibration_curve.png`) montre que le score brut
surestime le risque et que la correction de prévalence le rapproche de la diagonale, avec un léger écart
dans les probabilités hautes. L'AUC par sous-groupe (âge, historique de blessures, sommeil) avec IC95 est
dans `models/artifacts/advanced_analysis.json` et l'onglet « Étude Scientifique ». Calculé sur données
**simulées** : cela ne prouve pas l'équité en conditions réelles.

## Données réelles (collecte terrain)

Le dashboard contient un journal de saisie quotidien (onglet "Données Réelles"). Chaque jour
l'athlète note : effort perçu de la séance (0-10), durée, heures de sommeil, FC au réveil, et
si une blessure est survenue. À partir de 14 jours de journal, ATHÉNA calcule les features du
modèle (charge = RPE × durée, méthode sRPE de Foster 2001) et le score de risque.

**Évaluation honnête** : `evaluer_sur_donnees_reelles` mesure l'AUC du modèle de production sur
les fenêtres « 14 jours d'historique -> blessure dans les 7 jours suivants », avec un IC95 par
bootstrap *par athlète* et deux baselines naïves (ACWR seul, sommeil seul). Elle refuse de
conclure tant qu'il n'y a pas au moins 5 athlètes et 15 fenêtres avec ET sans blessure.
Si l'IC95 inclut 0.5, le résultat est rapporté tel quel : le modèle n'est pas démontré meilleur
que le hasard sur ces données. Test du pipeline (données factices, jamais mélangées au journal
réel) : `python -m models.test_donnees_reelles`.

Features que le journal ne mesure pas (lactate, bilan calorique, et âge/taille si non fournis) :
valeur de référence du jeu d'entraînement, signalée comme « non mesurée » dans l'interface.
Import d'un jeu de données public : `adapter_csv_externe(chemin, correspondance)`.

Éthique : identifiants pseudonymisés, consentement écrit de chaque athlète (et du tuteur si mineur).

## Test en direct sur un CSV inédit (jour de la finale)
`models/testeur_csv.py` évalue le modèle sur un fichier jamais vu, sans rien deviner en silence : il affiche
l'association colonnes du fichier -> 10 variables du modèle (noms exacts, synonymes FR/EN, approximation prudente, ou
`--mapping`), signale les variables absentes, applique le même nettoyage qu'en production (manquant -> milieu du domaine,
hors domaine -> borné, avec compteurs), puis donne ROC-AUC avec IC95 bootstrap, précision/rappel, rappel parmi les 20 %
les plus à risque, Brier et fiabilité par tranches (et MAE/RMSE/R² si une colonne fatigue existe). Il lit `;` ou `,`,
la virgule décimale et les accents (UTF-8 ou Latin-1), et refuse proprement une étiquette illisible ou un fichier sans
aucune colonne reconnue (code de sortie 2, message en français).

```bash
python -m models.testeur_csv --demo                            # répétition générale (jeu inédit simulé, colonnes renommées)
python -m models.testeur_csv --test inedit.csv                 # détection automatique des colonnes
python -m models.testeur_csv --test inedit.csv --cible blessure --mapping mon_mapping.json
python -m models.testeur_csv --train train.csv --test inedit.csv   # réentraîne GB d=1, d=2, d=3, régression logistique
python -m models.test_testeur_csv                              # tests de l'outil
```
Sorties dans `sorties_test/<horodatage>/` : `predictions.csv`, `rapport.md`, `resume.json`.
Au premier lancement sur une machine, si les fichiers `.joblib` ne sont pas lisibles (autre version de scikit-learn), les
modèles sont réentraînés automatiquement (quelques secondes, résultat identique) : lancer `--demo` une fois avant la finale.

## Tests

```
python -m models.test_robustesse      # validation d'entrée, imputation, bruit
python -m models.test_donnees_reelles # pipeline données réelles (données factices, jamais mélangées au journal)
python -m models.test_sequence        # analyse de séquence vidéo (séries d'angles fabriquées)
python -m models.test_demo            # scénario de démo guidée + PDF
python -m models.test_testeur_csv     # outil de test en direct sur CSV inédit
python verifier_demo.py               # check-list globale avant démo
```

## Ce qui reste simulé

- Toutes les constantes vitales (FC, ECG, SpO2, température, lactate, glucose) : simulateur
  physiologique réaliste, en attente de capteurs Bluetooth/I2C (points d'injection prévus).
- Les profils de l'équipe nationale (mode "National Team") sont fictifs, pour la
  démonstration.

## Pistes d'amélioration

1. Enregistrer la vidéo de démonstration (plan minuté dans `docs/DEMO_ET_PITCH.md`) ; l'onglet « Démo
   Guidée » déroule le scénario de façon identique à chaque fois.
2. Collecter le journal sur plusieurs athlètes pendant plusieurs semaines (onglet « Données Réelles ») et
   rapporter l'AUC réelle, même modeste, avec son intervalle de confiance.
3. Valider l'analyse de séquence sur de vraies vidéos filmées par l'équipe (voir limites ci-dessus).
4. Trouver un jeu public quotidien de blessures et l'évaluer avec `data/importer_public.py`.
