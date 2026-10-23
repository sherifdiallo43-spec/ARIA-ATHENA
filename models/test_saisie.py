# ATHÉNA - models/test_saisie.py
# Vérifie l'ENREGISTREMENT des données : journal, séances (ACWR) et repas.
# Les fichiers sont redirigés vers un dossier temporaire : data/real/ n'est jamais touché.
#
# Lancement : python -m models.test_saisie  (depuis la racine du projet)

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data.donnees_reelles as D  # noqa: E402
from analytics.metrics import calculer_acwr  # noqa: E402


def _verifier(nom: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'ÉCHEC'}] {nom}")
    return bool(cond)


def main():
    dossier = Path(tempfile.mkdtemp(prefix="athena_saisie_"))
    vrais = D.CHEMIN_JOURNAL, D.CHEMIN_REPAS
    D.CHEMIN_JOURNAL, D.CHEMIN_REPAS = dossier / "journal.csv", dossier / "repas.csv"
    ok = True
    try:
        # 1. Un identifiant « 007 » reste « 007 » après relecture (pas de 7 en double).
        D.ajouter_entree({"athlete_id": "007", "date": "2026-10-20", "rpe": 5, "duree_min": 60})
        D.ajouter_entree({"athlete_id": "007", "date": "2026-10-21", "rpe": 5, "duree_min": 60})
        ok &= _verifier("identifiant « 007 » conservé", D.charger_journal()["athlete_id"].unique().tolist() == ["007"])

        # 2. Même athlète + même date : la journée est remplacée ET c'est signalé.
        _, statut = D.ajouter_entree({"athlete_id": " 007 ", "date": "21/10/2026", "rpe": 8, "duree_min": 30})
        j = D.charger_journal()
        ok &= _verifier("même date remplacée et signalée",
                        statut == "remplacee" and len(j) == 2 and j["rpe"].tolist() == [5, 8])

        # 3. Deux séances le même jour s'additionnent sans effacer le sommeil.
        D.ajouter_entree({"athlete_id": "A01", "date": "2026-10-22", "rpe": 6, "duree_min": 60, "sommeil_h": 7.5})
        _, info = D.ajouter_seance("a01", "2026-10-22", 8, 30)
        ligne = D.charger_journal().query("athlete_id == 'A01'").iloc[0]
        ok &= _verifier("2 séances le même jour cumulées",
                        info["cumulee"] and info["charge_jour"] == 600 and ligne["duree_min"] == 90
                        and round(ligne["rpe"] * ligne["duree_min"]) == 600 and ligne["sommeil_h"] == 7.5)

        # 4. Identifiant vide refusé avec un message clair (au lieu d'un faux « enregistré »).
        try:
            D.ajouter_entree({"athlete_id": "  ", "date": "2026-10-22"})
            refuse = False
        except D.ErreurJournal:
            refuse = True
        ok &= _verifier("identifiant vide refusé", refuse)

        # 5. Identifiant manquant dans un fichier importé : ligne écartée, pas de plantage (pandas 3).
        df, _ = D.preparer_journal(pd.DataFrame({"athlete_id": ["A01", None, float("nan")],
                                                 "date": ["2026-10-01"] * 3}))
        ok &= _verifier("identifiant NaN écarté", df["athlete_id"].tolist() == ["A01"]
                        and sorted(df["athlete_id"].unique()) == ["A01"])

        # 6. Dates françaises JJ/MM/AAAA lues jour d'abord.
        dates = D.lire_dates(["01/03/2026", "23/03/2026", "2026-03-05"]).dt.strftime("%Y-%m-%d").tolist()
        ok &= _verifier("dates JJ/MM/AAAA", dates == ["2026-03-01", "2026-03-23", "2026-03-05"])

        # 7. Journal réenregistré par Excel (« ; », latin-1) : relu sans perte.
        D.CHEMIN_JOURNAL.write_bytes("athlete_id;date;rpe;duree_min;sommeil_h;fc_repos;poids_kg;douleur;blessure\n"
                                     "Élan;01/03/2026;5;60;7,5;60;;;0\nÉlan;02/03/2026;6;45;7;61;;;1\n"
                                     .encode("latin-1"))
        j = D.charger_journal()
        ok &= _verifier("journal Excel (« ; », latin-1) relu", len(j) == 2 and j["sommeil_h"].tolist() == [7.5, 7.0])

        # 8. Fichier illisible : erreur claire et fichier NON écrasé par la sauvegarde suivante.
        D.CHEMIN_JOURNAL.write_text("athlete_id,date\n,pas une date\n", encoding="utf-8")
        avant = D.CHEMIN_JOURNAL.read_bytes()
        try:
            D.ajouter_entree({"athlete_id": "A01", "date": "2026-10-22"})
            protege = False
        except D.ErreurJournal:
            protege = D.CHEMIN_JOURNAL.read_bytes() == avant
        ok &= _verifier("journal illisible jamais écrasé", protege)
        D.CHEMIN_JOURNAL.unlink()

        # 9. Repas : nom facultatif, relu par athlète et par jour, suppression, kcal = 0 refusé.
        D.ajouter_repas("A01", "2026-10-23", "08:00", "", 600, 30)
        D.ajouter_repas("A01", "2026-10-23", "12:30", "Déjeuner", 800)
        D.ajouter_repas("A02", "2026-10-23", "12:30", "Déjeuner", 500)
        D.ajouter_repas("A01", "2026-10-22", "12:30", "Déjeuner", 700)
        r = D.charger_repas("a01", "2026-10-23")
        ok &= _verifier("repas relus par athlète et par jour",
                        r["kcal"].sum() == 1400 and r["repas"].iloc[0] == "Repas de 08:00")
        D.supprimer_repas(r["id"].iloc[0])
        ok &= _verifier("repas supprimé", D.charger_repas("A01", "2026-10-23")["kcal"].sum() == 800)
        try:
            D.ajouter_repas("A01", "2026-10-23", "13:00", "Vide", 0)
            refuse = False
        except D.ErreurJournal:
            refuse = True
        ok &= _verifier("repas à 0 kcal refusé", refuse)

        # 10. ACWR par jour calendaire : 3 séances le même jour font MONTER l'ACWR
        #     (avant : chaque ligne comptait comme un jour et l'ACWR baissait).
        jours = pd.date_range(end="2026-10-23", periods=28).date
        base = pd.DataFrame({"date": jours, "charge": [400] * 28})
        plus = pd.concat([base, pd.DataFrame({"date": [jours[-1]] * 3, "charge": [400] * 3})], ignore_index=True)
        a0, a1 = calculer_acwr(base)["acwr"], calculer_acwr(plus)["acwr"]
        ok &= _verifier(f"ACWR par jour calendaire ({a0} -> {a1})", a0 == 1.0 and a1 > a0)

        # 11. Charge réelle sur 28 jours : jours sans saisie = repos (0), même ACWR que le journal.
        D.ajouter_entree({"athlete_id": "B01", "date": "2026-10-01", "rpe": 5, "duree_min": 60})
        D.ajouter_entree({"athlete_id": "B01", "date": "2026-10-10", "rpe": 5, "duree_min": 60})
        ch = D.charge_28j(D.charger_journal(), "B01", fin="2026-10-12")
        ok &= _verifier("charge réelle jour par jour", len(ch) == 12 and ch["charge"].sum() == 600
                        and (ch["charge"] == 0).sum() == 10)
    finally:
        D.CHEMIN_JOURNAL, D.CHEMIN_REPAS = vrais
        shutil.rmtree(dossier, ignore_errors=True)

    print("\nTous les tests passent." if ok else "\nDes tests ont échoué.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
