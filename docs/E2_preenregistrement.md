# Expérience E2 — hypothèses et règles de décision (fixées AVANT exécution)

Fichier écrit le 2026-10-10 à 11:41 UTC, avant toute exécution de `models/experiment_profondeur.py`.
Motivation : l'expérience E1 (familles de modèles) est restée non concluante — l'IC95 de la différence
d'AUC entre Gradient Boosting et régression logistique contient 0. E2 cherche à expliquer pourquoi.

## Propriété du jeu de données utilisée
Dans le générateur (`models/train_injury_model.py::_generer_dataset`), le risque latent est une SOMME de
termes univariés (fonctions seuil de l'ACWR, du sommeil, de l'intensité, du déficit calorique, etc.) plus un
bruit gaussien non observé (écart-type 0,9) ; il n'y a aucun terme croisé entre variables. Deux variables
(charge chronique, FC de repos) n'interviennent pas du tout.

## Propriété du modèle utilisée
Un arbre de profondeur d modélise des interactions d'ordre au plus d. Profondeur 1 (souche) = modèle additif.
La régression logistique est additive ET linéaire en les variables brutes.

## Conditions
- Variable indépendante : max_depth du Gradient Boosting ∈ {1, 2, 3, 4, 6} (5 conditions).
- Variables contrôlées : learning_rate = 0,08 ; n_estimators = 180 (configuration de référence de E1) ;
  pondération « balanced » ; mêmes données et mêmes partitions pour toutes les conditions.
  Variante E2b : arrêt précoce (n_iter_no_change = 10, validation_fraction = 0,15, ≤ 600 arbres) pour que
  chaque profondeur ait un budget d'arbres adapté.
- Références : régression logistique (E1), et « oracle » = AUC du classement par le risque latent SANS bruit
  (optimal au sens de l'AUC : la probabilité de blessure sachant X est monotone en ce risque).
- Métrique : ROC-AUC sur jeu de test. Résultat primaire : P2 = 10 jeux générés indépendamment
  (graines de génération 1000 à 1009, n = 6000, partition 80/20). P1 = protocole E1 (jeu graine 42,
  10 partitions) rapporté comme contrôle de cohérence, sans servir à décider (partitions non indépendantes).
- Intervalles : IC95 par loi de Student (9 ddl) sur les différences appariées entre les 10 jeux.

## Hypothèses et règles de décision (sur P2)
- **H2 — les interactions n'apportent rien.** Prédiction : pour chaque d ∈ {2, 3, 4, 6}, la borne haute de l'IC95
  de ΔAUC(d − 1) est < +0,010. Confirmée si les 4 bornes hautes sont < +0,010 ; contredite si, pour au moins un d,
  la borne basse est > 0 et ΔAUC ≥ +0,010 ; sinon non concluante.
- **H3 — le plafond est le bruit, pas le modèle.** Prédiction : l'écart moyen entre l'AUC oracle et l'AUC de la
  meilleure condition de profondeur est < 0,030. Confirmée si écart < 0,030 ; contredite sinon.
- **H4 — la non-linéarité univariée explique l'avantage sur la régression logistique.** Prédiction : ΔAUC(profondeur 1 −
  régression logistique) a une borne basse d'IC95 > 0. Confirmée si oui ; contredite si la borne haute est < 0 ;
  sinon non concluante.

## Analyses descriptives (sans hypothèse)
Plancher de bruit pour la fatigue (MAE/R² du prédicteur qui connaît la partie déterministe du générateur) ;
corrélation de rang entre l'importance par permutation du modèle et celle de l'oracle.

## Prédictions personnelles consignées avant exécution (non contraignantes)
Je m'attends à H2 confirmée, H3 proche du seuil (l'écart pourrait dépasser 0,03), H4 plausible mais incertaine.
