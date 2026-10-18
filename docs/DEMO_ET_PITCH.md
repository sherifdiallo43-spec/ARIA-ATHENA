# ATHÉNA — Démo vidéo (2 min) et pitch

## Vidéo de démo — plan minuté
1. **0:00–0:15 — Le problème.** « Les blessures coûtent des saisons aux athlètes africains, et les outils de suivi coûtent des milliers de dollars. ATHÉNA tourne sur un simple ordinateur, hors-ligne. »
2. **0:15–0:40 — Alerte.** Onglet IA Prédictive : choisir un athlète à risque élevé, montrer le score et les facteurs clés.
3. **0:40–1:05 — What-if.** Simulateur : « +1 h de sommeil » → montrer la baisse de risque. Dire que c'est calculé par le modèle, pas une règle inventée.
4. **1:05–1:25 — AI Coach + PDF.** Recommandations priorisées, clic sur « Generate AI Report ».
5. **1:25–1:50 — Le terrain.** Onglet Données Réelles : saisir une journée, montrer le score calculé sur données réelles et le message honnête « échantillon insuffisant, continuer la collecte » (ou l'AUC avec son IC95 si assez de données).
6. **1:50–2:00 — Conclusion.** « Un outil sobre, explicable, honnête sur ses limites, prêt pour des capteurs réels. »

Astuce : l'onglet « Démo Guidée » déroule les étapes 2 à 4 de façon identique à chaque fois, sans internet ; l'utiliser comme fil conducteur.

Conseils : enregistrer en une prise après répétition ; avoir le dashboard déjà lancé ; modèle de pose déjà téléchargé (`--setup`).

## Pitch oral — 3 minutes
- **Problème (30 s)** : blessures évitables, peu de moyens, pas d'outils adaptés.
- **Solution (45 s)** : score de risque, fatigue, ACWR, coach IA, analyse posturale — dans un seul tableau de bord.
- **Preuve scientifique (60 s)** : « Première expérience : le Gradient Boosting ne bat pas clairement la régression logistique, IC95 de la différence contenant 0. Au lieu de nous arrêter là, nous avons posé une question : quelle part de l'erreur vient du modèle, quelle part du bruit ? Nous avons fait varier un seul facteur, la profondeur des arbres, avec des hypothèses écrites avant l'exécution, et comparé à un oracle, le meilleur AUC possible. Résultat : un modèle additif (profondeur 1) est à 0,007 de l'oracle ; chaque niveau d'interaction retire de l'AUC. Le plafond de 0,70 est celui du bruit, pas de notre modèle. Puis nous avons testé le cas inverse : si les données contiennent une interaction, la profondeur 2 devient la meilleure ; c'est pourquoi notre modèle de production est de profondeur 2, le choix le plus robuste. » (Figures 4 et 7 du rapport.)
- **Pont vers le réel (30 s)** : journal quotidien, évaluation par bootstrap par athlète, refus de conclure sur un petit échantillon.
- **Impact et suite (15 s)** : coût zéro, hors-ligne, français/anglais ; prochaine étape = capteurs Bluetooth/I2C.

## Questions pièges — réponses courtes
- **Pourquoi des données synthétiques ?** Aucun jeu de données labellisé africain n'est ouvert. Le simulateur suit la littérature (ACWR, sommeil, historique) ; le journal réel sert à valider hors simulation.
- **Votre AUC est modeste (0.68) ?** Oui, et c'est cohérent avec la littérature sur la prédiction de blessure (Leckey et al., 2025). Nous l'utilisons comme aide à la décision, pas comme diagnostic ; le rappel est ~50 %.
- **Pourquoi Gradient Boosting si Random Forest a une meilleure AUC ?** Écart non significatif (IC95 de la différence inclut 0). Surtout : E2 montre que la profondeur compte plus que la famille ; un GB de profondeur 1 est à 0,007 de l'AUC maximale possible sur nos données, et il bat la régression logistique de +0,020 (IC95 [0,012 ; 0,028]).
- **Et si une mesure manque ?** Validation d'entrée : imputation + clipping, rapport de qualité, test de robustesse (jusqu'à 50 % de valeurs manquantes).
- **Pourquoi 20 % de prévalence ?** Ordre de grandeur de la littérature ; le score affiché est relatif, la probabilité absolue corrigée est disponible.
- **Le modèle est-il équitable entre groupes ?** Sur la simulation, l'AUC est similaire entre âges, avec ou sans historique de blessure (IC95 dans l'onglet Étude Scientifique) ; ce n'est pas une preuve en conditions réelles, d'où la collecte terrain.
- **Vos probabilités sont-elles fiables ?** Courbe de fiabilité : le score brut surestime, la correction de prévalence corrige l'essentiel, il reste un écart aux probabilités hautes. On l'utilise par tranches (faible/modéré/élevé), pas comme probabilité exacte.
- **L'analyse vidéo est-elle précise ?** C'est un démonstrateur 2D : il détecte les déséquilibres grossiers (amplitude, symétrie, cadence), pas un diagnostic. Il refuse de conclure si la silhouette est mal détectée.
- **Données personnelles ?** Identifiants pseudonymisés, consentement écrit, stockage local.

- **Votre profondeur 1 gagne parce que vos données sont additives ?** Oui, c'est ce que nous disons : notre générateur n'a pas de terme croisé. Nous avons testé le cas inverse (E3, terme croisé ACWR × sommeil) : avec une interaction forte, la profondeur 2 devient meilleure de +0,044 d'AUC. La leçon transférable est d'adapter la structure du modèle à celle des données et de se comparer à un plafond.
- **Pourquoi votre modèle de production est-il de profondeur 2 et pas 1 ?** Parce que nous ne savons pas si les données inconnues ont des interactions. E3 : sur un risque additif, la profondeur 2 perd au plus 0,010 d'AUC ; si une interaction forte existe, elle gagne 0,044 alors que la profondeur 1 perd 0,044. C'est le choix qui minimise le pire regret. Limite honnête : une interaction modérée (c = 0,5) n'est pas détectable avec 6 000 exemples (hypothèse H5 non concluante).

## Jour J — checklist test sur données inédites
1. La veille, sur l'ordinateur de la finale : `python -m models.testeur_csv --demo` (doit se terminer sans erreur et afficher ROC-AUC ≈ 0,70).
2. Copier le fichier du jury dans le dossier du projet ; lancer `python -m models.testeur_csv --test fichier.csv`.
3. Lire d'abord le tableau « Correspondance des colonnes » : si une variable est mal associée ou absente, relancer avec `--mapping` (JSON {variable_modèle: colonne}).
4. Si le jury fournit aussi un jeu d'entraînement : ajouter `--train train.csv` ; comparer GB d=1, GB d=3 et régression logistique.
5. Montrer le rapport (`rapport.md`) et dire à voix haute les limites affichées (échantillon petit, variables absentes).
