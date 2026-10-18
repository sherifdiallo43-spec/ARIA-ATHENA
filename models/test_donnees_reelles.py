# ATHÉNA - models/test_donnees_reelles.py
# Vérifie que le pipeline "données réelles" fonctionne de bout en bout.
# ATTENTION : les données utilisées ici sont FABRIQUÉES, uniquement pour tester le code.
# Elles ne sont jamais écrites dans data/real/journal.csv et n'ont aucune valeur scientifique.
#
# Lancement : python -m models.test_donnees_reelles  (depuis la racine du projet)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.donnees_reelles import (  # noqa: E402
    preparer_journal, risque_actuel, evaluer_sur_donnees_reelles, gabarit_csv, COLONNES,
)


def _journal_factice(n_athletes: int, n_jours: int, graine: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(graine)
    lignes = []
    for a in range(n_athletes):
        debut = pd.Timestamp("2026-01-01")
        for j in range(n_jours):
            seance = rng.random() < 0.7
            lignes.append({
                "athlete_id": f"T{a:02d}", "date": debut + pd.Timedelta(days=j),
                "rpe": rng.integers(3, 9) if seance else 0,
                "duree_min": rng.integers(30, 100) if seance else 0,
                "sommeil_h": rng.normal(7, 1), "fc_repos": rng.normal(60, 4),
                "poids_kg": 70, "douleur": rng.integers(0, 4),
                "blessure": int(rng.random() < 0.02),
            })
    return pd.DataFrame(lignes)


def main():
    ok = True

    # 1. Nettoyage : valeurs aberrantes, doublons, dates invalides
    sale = pd.DataFrame([
        {"athlete_id": "A01", "date": "2026-03-01", "rpe": 55, "duree_min": 60, "sommeil_h": 7},
        {"athlete_id": "A01", "date": "2026-03-01", "rpe": 6, "duree_min": 60, "sommeil_h": 7},
        {"athlete_id": "A01", "date": "pas-une-date", "rpe": 6},
    ])
    propre, rap = preparer_journal(sale)
    cond = rap["doublons_supprimes"] == 1 and rap["dates_invalides"] == 1 and len(propre) == 1
    print(f"[{'OK' if cond else 'ÉCHEC'}] nettoyage : {rap}")
    ok &= cond

    # 2. Journal trop court -> refus explicite de calculer
    court, _ = preparer_journal(_journal_factice(1, 5))
    r = risque_actuel(court, "T00")
    cond = r["statut"] == "historique_insuffisant"
    print(f"[{'OK' if cond else 'ÉCHEC'}] historique insuffisant détecté")
    ok &= cond

    # 3. Journal suffisant -> un score de risque
    long_, _ = preparer_journal(_journal_factice(1, 40))
    r = risque_actuel(long_, "T00")
    cond = r["statut"] == "ok" and 0 <= r["profil"]["injury_risk_pct"] <= 100
    print(f"[{'OK' if cond else 'ÉCHEC'}] score calculé : {r['profil']['injury_risk_pct']} % "
          f"(non mesurés : {r['info']['non_mesures']})")
    ok &= cond

    # 4. Évaluation : petit échantillon -> doit REFUSER de conclure
    petit, _ = preparer_journal(_journal_factice(2, 30))
    ev = evaluer_sur_donnees_reelles(petit)
    cond = ev["statut"] == "insuffisant"
    print(f"[{'OK' if cond else 'ÉCHEC'}] petit échantillon -> {ev['statut']}")
    ok &= cond

    # 5. Évaluation : grand échantillon -> métriques + IC (sur données factices : valeurs sans sens)
    grand, _ = preparer_journal(_journal_factice(12, 90, graine=3))
    ev = evaluer_sur_donnees_reelles(grand, n_bootstrap=200)
    cond = ev["statut"] in ("ok", "insuffisant")
    print(f"[{'OK' if cond else 'ÉCHEC'}] grand échantillon -> {ev['statut']} "
          f"({ev['n_athletes']} athlètes, {ev['n_positifs']} positifs) "
          f"{'AUC=' + str(ev.get('auc_modele')) + ' IC95=' + str(ev.get('ic95_auc')) if ev['statut'] == 'ok' else ''}")
    ok &= cond

    # 5b. Données hebdomadaires -> refus (les jours manquants seraient comptés comme repos)
    hebdo = _journal_factice(10, 120)
    hebdo = hebdo[hebdo["date"].dt.dayofweek == 0]
    hebdo, _ = preparer_journal(hebdo)
    ev = evaluer_sur_donnees_reelles(hebdo)
    cond = ev["statut"] == "resolution_inadaptee"
    print(f"[{'OK' if cond else 'ÉCHEC'}] données hebdomadaires -> {ev['statut']}")
    ok &= cond

    # 6. Gabarit
    cond = gabarit_csv().strip().split(",") == COLONNES
    print(f"[{'OK' if cond else 'ÉCHEC'}] gabarit CSV")
    ok &= cond

    print("\nTous les tests passent." if ok else "\nDes tests ont échoué.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
