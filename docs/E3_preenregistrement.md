# Expérience E3 — hypothèses et règles de décision (fixées AVANT exécution)

Fichier écrit le 2026-10-18 17:31 UTC, avant toute exécution de `models/experiment_interaction.py`.
Motivation : E2 montre que la profondeur 1 est optimale parce que le générateur est additif. Que se passe-t-il quand
les données contiennent une interaction ? Un jeu de test inconnu (jury) peut être plus complexe que le nôtre : on veut
savoir quelle profondeur est un choix **robuste**, pas seulement optimal dans un cas.

## Variante du générateur
Même variables X que `_generer_dataset`. Risque latent r = r_additif (identique à E2) + c · zA · zS, avec
zA = (ACWR − 1,05) / 0,35 et zS = (7,0 − sommeil) / 1,1 (deux variables d'écart-type ≈ 1, indépendantes : le terme croisé a donc
un écart-type ≈ c, contre ≈ 0,86 pour le risque additif et 0,9 pour le bruit). Étiquette : y ~ Bernoulli(σ(r + ε − b)),
ε ~ N(0 ; 0,9²) non observé ; l'ordonnée b est choisie par jeu pour que la prévalence attendue soit 20 %, comme en E1/E2.
Un terme croisé bilinéaire est une interaction d'ordre exactement 2 : un arbre de profondeur 2 peut la représenter.
Oracle = AUC du classement par r (sans bruit), optimal au sens de l'AUC.

## Conditions
- Variable indépendante 1 : force de l'interaction c ∈ {0 ; 0,25 ; 0,5 ; 1,0} (4 conditions ; c = 0 reproduit E2).
- Variable indépendante 2 : profondeur max du Gradient Boosting d ∈ {1, 2, 3, 4, 6}.
- Contrôlées : ν = 0,08 ; 180 arbres (variante : arrêt précoce, comme E2) ; pondération « balanced » ; mêmes partitions.
- Références : régression logistique et oracle. Métrique : ROC-AUC sur test.
- Protocole : pour chaque c, 10 jeux générés indépendamment (graines 2000 à 2009, n = 6000), partition 80/20 ;
  IC95 de Student (9 ddl) sur différences appariées. Les décisions portent sur la condition « 180 arbres ».

## Hypothèses et règles de décision
- **H5 — l'interaction fait monter la profondeur optimale.** Prédiction : pour c = 0,5 et c = 1,0, la borne basse de l'IC95 de
  ΔAUC(d = 2 − d = 1) est > +0,005. Confirmée si les deux conditions sont vérifiées ; contredite si, pour c = 1,0, la borne haute est
  < +0,005 ; sinon non concluante.
- **H6 — rien à gagner au-delà de l'ordre réel de l'interaction.** Prédiction : pour chaque c, la borne haute de l'IC95 de
  ΔAUC(d = 3 − d = 2) est < +0,010. Confirmée si les 4 bornes sont < 0,010 ; contredite si, pour un c, la borne basse est > 0 et la différence ≥ +0,010 ;
  sinon non concluante.
- **H7 — la profondeur 2 est un choix robuste.** Le regret de d pour un c = AUC moyen du meilleur d (parmi 1, 2, 3, 4, 6) − AUC moyen de d.
  Prédiction : le regret maximal de d = 2 sur les 4 valeurs de c est < 0,012 ET inférieur au regret maximal de d = 1.
  Confirmée si les deux conditions sont vérifiées ; contredite si le regret maximal de d = 2 est ≥ celui de d = 1 ; sinon non concluante.
- **H8 — réplication de E2.** Pour c = 0, la borne haute de l'IC95 de ΔAUC(d = 2 − d = 1) est < +0,010 (nouvelles graines).
  Contrôle de cohérence : la prévalence moyenne du test doit rester entre 0,15 et 0,25 pour toutes les conditions.

## Prédictions personnelles consignées avant exécution (non contraignantes)
H8 et H6 probables ; H5 sûre pour c = 1,0 mais incertaine pour c = 0,5 ; H7 plausible. Je n'ai pas d'attente chiffrée sur le regret de d = 1.
