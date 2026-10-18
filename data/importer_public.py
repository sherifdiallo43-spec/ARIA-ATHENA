# ATHÉNA - data/importer_public.py
# Convertit un jeu de données PUBLIC de blessures sportives vers le format du journal ATHÉNA,
# pour l'évaluer avec la même méthode que les données saisies dans l'application.
#
# Ce script ne devine AUCUN format : il faut fournir le fichier ET la correspondance des colonnes.
#
# Usage :
#   python -m data.importer_public --fichier mon_jeu.csv --mapping mon_mapping.json --nom coureurs
#   -> écrit data/real/public_coureurs.csv, évaluable dans l'onglet "Données Réelles".
#
# Format du mapping : {"colonne_athena": "colonne_du_fichier_source", ...}. Voir data/mapping_exemple.json
# (EXEMPLE avec des noms de colonnes fictifs — à remplacer par ceux du vrai fichier).
#
# IMPORTANT — résolution temporelle : l'évaluation exige des données QUOTIDIENNES (≥ 60 % des jours
# renseignés). Beaucoup de jeux publics sont hebdomadaires : le script le détecte et prévient.
# Où chercher : portails de données de recherche (DataverseNL, Zenodo, Figshare, Kaggle) avec les
# mots-clés "injury prediction", "training load", "runners", "athletes daily log". Vérifier la licence
# (citer la source) et que les données sont anonymisées avant usage.

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.donnees_reelles import (  # noqa: E402
    DOSSIER_REEL, COLONNES, adapter_csv_externe, taux_saisie_global,
)

AIDE = """
Pour utiliser un jeu public :
  1. Télécharger le CSV depuis la source (vérifier la licence).
  2. Créer un fichier mapping.json associant NOS colonnes aux siennes (voir data/mapping_exemple.json).
     Colonnes ATHÉNA : {colonnes}
     Obligatoires : athlete_id, date, blessure. Fortement utiles : rpe, duree_min (ou charge), sommeil_h, fc_repos.
  3. Relancer : python -m data.importer_public --fichier <csv> --mapping <json> --nom <nom_court>
"""


def importer(fichier: Path, mapping_json: Path, nom: str) -> Path:
    if not fichier.exists():
        raise FileNotFoundError(f"Fichier introuvable : {fichier}\n" + AIDE.format(colonnes=", ".join(COLONNES)))
    mapping = json.loads(mapping_json.read_text(encoding="utf-8"))
    for oblig in ("athlete_id", "date", "blessure"):
        if oblig not in mapping:
            raise ValueError(f"Le mapping doit définir '{oblig}'.")
    df, rapport = adapter_csv_externe(fichier, mapping)
    taux = taux_saisie_global(df)
    sortie = DOSSIER_REEL / f"public_{nom}.csv"
    df.to_csv(sortie, index=False)
    print(f"{rapport['lignes_valides']} lignes valides, {df['athlete_id'].nunique()} athlètes -> {sortie}")
    print(f"Valeurs hors plage ignorées : {rapport['valeurs_hors_plage']} ; doublons : {rapport['doublons_supprimes']}")
    if rapport["colonnes_manquantes"]:
        print("Colonnes absentes du jeu (valeurs de référence utilisées) :", ", ".join(rapport["colonnes_manquantes"]))
    print(f"Taux de jours renseignés : {taux:.0%}")
    if taux < 0.6:
        print("⚠️  Données non quotidiennes : l'évaluation sera refusée (voir en-tête du script).")
    return sortie


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Importer un jeu de données public vers le format ATHÉNA")
    ap.add_argument("--fichier", type=Path)
    ap.add_argument("--mapping", type=Path)
    ap.add_argument("--nom", default="jeu_public")
    a = ap.parse_args()
    if not a.fichier or not a.mapping:
        print(AIDE.format(colonnes=", ".join(COLONNES)))
        sys.exit(0)
    importer(a.fichier, a.mapping, a.nom)
