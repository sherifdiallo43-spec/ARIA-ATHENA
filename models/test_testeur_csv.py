# Tests de l'outil de test en direct (models/testeur_csv.py). Lancement : python -m models.test_testeur_csv
import atexit
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .injury_risk import construire_vecteur_features, valider_et_nettoyer_features
from .train_injury_model import FEATURE_NAMES, _generer_dataset
from . import testeur_csv as T


def _tmp() -> Path:
    d = Path(tempfile.mkdtemp(prefix="athena_test_"))
    atexit.register(shutil.rmtree, d, ignore_errors=True)   # pas de dossiers laissés dans %TEMP%
    return d


def test_repetition_generale():
    d = _tmp()
    chemin = T.jeu_de_repetition(d, n=2000, graine=321)
    res = T.analyser(chemin)
    assert res["variables_mesurees"] == 10, res["correspondance"]
    auc = res["production"]["auc"]
    assert 0.62 < auc < 0.78, f"AUC hors plage attendue : {auc}"
    assert res["production"]["ic95"][0] < auc < res["production"]["ic95"][1]
    assert res["fatigue"]["mae"] < 1.0
    assert res["qualite"]["sommeil_h"]["sur_domaine"] >= 1          # les 25 h injectées sont bornées
    dossier = T.ecrire_sorties(res, d)
    for f in ("predictions.csv", "rapport.md", "resume.json"):
        assert (dossier / f).exists(), f
    pred = pd.read_csv(dossier / "predictions.csv")
    assert len(pred) == 2000 and pred["athena_score_risque"].between(0, 1).all()
    print(f"[OK] répétition générale : AUC {auc}, 10/10 colonnes associées")


def test_nettoyage_identique_a_la_production():
    X, _, _ = _generer_dataset(n=300, graine=5)
    X = X.copy()
    X[3, 0] = np.nan
    X[7, 3] = 30.0
    X[9, 4] = -5
    Xn, _ = T.nettoyer_matrice(X)
    for i in (3, 7, 9, 0):
        ref, _ = valider_et_nettoyer_features(X[i])
        assert np.allclose(ref[0], Xn[i]), (i, ref[0], Xn[i])
    print("[OK] nettoyage vectorisé identique à valider_et_nettoyer_features")


def test_format_exotique_et_sans_etiquette():
    d = _tmp()
    X, y, _ = _generer_dataset(n=400, graine=9)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df = df.rename(columns={"sommeil_h": "Heures de sommeil", "acwr": "ACWR", "age": "Âge"})
    df["extra"] = "texte"
    chemin = d / "x.csv"
    df.to_csv(chemin, sep=";", decimal=",", index=False, encoding="latin-1")
    res = T.analyser(chemin)
    assert "production" not in res and res["variables_mesurees"] == 10
    assert res["correspondance"]["sommeil_h"]["colonne"] == "Heures de sommeil"
    assert res["correspondance"]["age"]["colonne"] == "Âge"
    print("[OK] latin-1, « ; », virgule décimale, colonne inutile, pas d'étiquette")


def test_variables_absentes_et_mapping():
    d = _tmp()
    X, y, _ = _generer_dataset(n=600, graine=11)
    df = pd.DataFrame({"sleep_quality_hours": X[:, 3], "load_ratio_x": X[:, 0], "inj": y})
    chemin = d / "p.csv"
    df.to_csv(chemin, index=False)
    res = T.analyser(chemin, cible="inj", mapping={"sommeil_h": "sleep_quality_hours", "acwr": "load_ratio_x"})
    assert res["variables_mesurees"] == 2
    assert any("absentes" in a for a in res["avertissements"]) and any("moins de 5" in a.lower() for a in res["avertissements"])
    assert res["production"]["auc"] > 0.55
    try:
        T.analyser(chemin, mapping={"sommeil_h": "colonne_inexistante"})
        raise AssertionError("un mapping vers une colonne absente doit échouer")
    except ValueError as e:
        assert "colonne_inexistante" in str(e)
    print("[OK] mapping explicite, variables absentes signalées, mapping invalide refusé")


def test_cas_degeneres():
    d = _tmp()
    X, y, _ = _generer_dataset(n=200, graine=2)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["blessure"] = 0
    df.to_csv(d / "une_classe.csv", index=False)
    res = T.analyser(d / "une_classe.csv")
    assert "auc" not in res["production"] and "avertissement" in res["production"]
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(d / "rien.csv", index=False)
    try:
        T.analyser(d / "rien.csv")
        raise AssertionError("aucune variable reconnue doit échouer proprement")
    except ValueError as e:
        assert "Aucune des 10 variables" in str(e)
    df2 = pd.DataFrame(X, columns=FEATURE_NAMES)
    df2["blessure"] = ["peut-etre"] * 200
    df2.to_csv(d / "etiq.csv", index=False)
    try:
        T.analyser(d / "etiq.csv")
        raise AssertionError("étiquette illisible doit échouer")
    except ValueError as e:
        assert "Étiquette" in str(e) or "étiquette" in str(e)
    try:
        T.analyser(d / "absent.csv")
        raise AssertionError
    except FileNotFoundError:
        pass
    print("[OK] une seule classe, aucune colonne reconnue, étiquette illisible, fichier absent")


def test_reentrainement():
    d = _tmp()
    X, y, _ = _generer_dataset(n=3000, graine=77)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["blessure"] = y
    df.iloc[:2000].to_csv(d / "train.csv", index=False)
    df.iloc[2000:].to_csv(d / "test.csv", index=False)
    res = T.analyser(d / "test.csv", chemin_train=d / "train.csv", n_boot=200)
    aucs = res["reentrainement"]["auc"]
    assert len(aucs) == 5 and all(0.55 < a["auc"] < 0.80 for a in aucs.values()), aucs
    print("[OK] réentraînement :", {k: v["auc"] for k, v in aucs.items()})


def test_ligne_de_commande():
    d = _tmp()
    code = T.main(["--demo", "--sortie", str(d)])
    assert code == 0
    assert T.main(["--test", str(d / "n_existe_pas.csv"), "--sortie", str(d)]) == 2
    print("[OK] ligne de commande : --demo et erreur propre (code 2)")


def test_verdict_et_imc_calcule():
    # Fichier de type « résultats olympiques » : âge, poids, taille, médaille, mais aucune variable clé.
    d = _tmp()
    rng = np.random.default_rng(7)
    n = 400
    jo = pd.DataFrame({"ID": range(n), "Name": [f"Athlete {i}" for i in range(n)], "Sex": "M",
                       "Age": rng.integers(18, 35, n), "Height": rng.integers(160, 200, n),
                       "Weight": rng.integers(55, 100, n), "Medal": rng.choice(["Gold", "NA"], n)})
    jo.to_csv(d / "jo.csv", index=False)
    res = T.analyser(d / "jo.csv")
    assert res["correspondance"]["imc"]["methode"].startswith("calculé"), res["correspondance"]["imc"]
    imc = pd.to_numeric(res["_predictions"][res["correspondance"]["imc"]["colonne"]])
    attendu = jo["Weight"] / (jo["Height"] / 100) ** 2
    assert np.allclose(imc, attendu.round(2)), "IMC mal calculé"
    assert res["verdict"]["statut"] == "incompatible", res["verdict"]
    assert "production" not in res                     # « Medal » n'est pas lue comme une blessure
    # Le jeu de répétition, lui, est compatible.
    rep = T.analyser(T.jeu_de_repetition(d, n=1500, graine=11))
    assert rep["verdict"]["statut"] == "compatible", rep["verdict"]
    # Données hors du domaine appris : verdict partiel, variable citée.
    X, y, _ = _generer_dataset(n=600, graine=3)
    hors = pd.DataFrame(X, columns=FEATURE_NAMES)
    hors["age"] = 60
    hors["blessure"] = y
    hors.to_csv(d / "hors.csv", index=False)
    v = T.analyser(d / "hors.csv")["verdict"]
    assert v["statut"] == "partiel" and "age" in v["hors_domaine"], v
    print("[OK] verdict incompatible / compatible / partiel, IMC calculé depuis poids et taille")


def test_unites_converties():
    # Sommeil en minutes, intensité sur 10, bilan calorique (mangé - besoin) : convertis et annoncés.
    d = _tmp()
    X, y, _ = _generer_dataset(n=800, graine=21)
    df =pd.DataFrame(X, columns=FEATURE_NAMES)
    df["blessure"] = y
    df.to_csv(d / "ref.csv", index=False)
    auc_ref = T.analyser(d / "ref.csv", n_boot=50)["production"]["auc"]
    autre = df.rename(columns={"sommeil_h": "sleep_minutes_x", "intensite_moyenne_7j": "intensity",
                               "deficit_calorique_pct": "calorie_balance"})
    autre["sleep_minutes_x"] = autre["sleep_minutes_x"] * 60
    autre["intensity"] = autre["intensity"] * 10
    autre["calorie_balance"] = -autre["calorie_balance"]
    autre.to_csv(d / "unites.csv", index=False)
    res = T.analyser(d / "unites.csv", mapping={"sommeil_h": "sleep_minutes_x"}, n_boot=50)
    texte = " ".join(res["avertissements"])
    assert "minutes" in texte and "0-10" in texte and "signe inversé" in texte, res["avertissements"]
    assert abs(res["production"]["auc"] - auc_ref) < 1e-9, (res["production"]["auc"], auc_ref)
    assert res["qualite"]["sommeil_h"]["sur_domaine"] == 0
    print(f"[OK] unités converties (minutes, échelle 0-10, bilan calorique) : même AUC {auc_ref}")


def test_etiquette_comptage_et_colonne_illisible():
    d = _tmp()
    X, y, _ = _generer_dataset(n=600, graine=31)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    rng = np.random.default_rng(0)
    df["nb_blessures_saison"] = np.where(y == 1, rng.integers(1, 4, len(y)), 0)   # 0, 1, 2, 3
    df["sommeil_h"] = "7h30"                                                       # illisible
    df.to_csv(d / "compte.csv", index=False)
    res = T.analyser(d / "compte.csv", cible="nb_blessures_saison", n_boot=50)
    assert "auc" in res["production"], res["production"]
    assert any("> 0" in a for a in res["avertissements"]), res["avertissements"]
    v = res["verdict"]
    assert "sommeil_h" in v["colonnes_vides"] and "sommeil_h" not in v["variables_cles_presentes"], v
    print(f"[OK] étiquette 0/1/2/3 lue comme blessure (> 0), colonne « 7h30 » comptée absente ({v['statut']})")


def test_rapport_console_windows():
    # Le rapport contient « ⚠️ » et « ² » : il ne doit pas planter sur une console cp1252.
    import contextlib
    import io
    d = _tmp()
    tampon = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with contextlib.redirect_stdout(tampon):
        code = T.main(["--demo", "--sortie", str(d)])
    assert code == 0
    print("[OK] rapport affiché sans erreur sur une console Windows (cp1252)")


def test_reponses_par_athlete():
    from .explicabilite import analyser_lot, expliquer_prediction, recommander_leviers
    d = _tmp()
    chemin = T.jeu_de_repetition(d, n=400, graine=77)
    res = T.analyser(chemin, n_boot=50)
    L, pa, s = res["_lisible"], res["par_athlete"], res["production"]["seuil_0_5"]
    assert len(L) == 400 and pa["colonne_identifiant"] == "player_id" and L["Athlète"].str.startswith("P").all()
    assert (L["Blessure prédite"] == "Oui").sum() == pa["blessures_predites"] == s["signales"]
    r = pa["resultats"]
    tp = r["Correct : blessure repérée"]
    assert abs(tp / (tp + r["Fausse alerte"]) - s["precision"]) < 1e-3, (r, s)
    assert abs(tp / (tp + r["Blessure manquée"]) - s["rappel"]) < 1e-3, (r, s)
    assert sum(r.values()) == 400

    # La version groupée donne les mêmes contributions et le même premier levier que la version individuelle.
    Xn, _ = T.nettoyer_matrice(res["_X"])
    lot = analyser_lot(Xn)
    for i in np.random.default_rng(0).choice(400, 5, replace=False):
        for e in expliquer_prediction(Xn[i:i + 1], top=9):
            assert abs(lot["contributions"][e["cle"]][i] - e["contribution_pts"]) <= 0.1, (i, e)
        leviers = recommander_leviers(Xn[i:i + 1])["leviers"]
        if leviers:
            meilleur = max(lot["gains_leviers"], key=lambda k: lot["gains_leviers"][k][i])
            assert meilleur == leviers[0]["cle"] and L.at[i, "Action prioritaire"].startswith(
                T.LEVIERS[meilleur][1]), (i, leviers[0], L.at[i, "Action prioritaire"])

    # Fichier Excel français : « ; », virgule décimale, accents intacts.
    dossier = T.ecrire_sorties(res, d)
    relu = pd.read_csv(dossier / "resultats_par_athlete.csv", sep=";", decimal=",", encoding="utf-8-sig")
    assert len(relu) == 400 and "Blessure prédite" in relu.columns and "Résultat" in relu.columns
    assert relu["Score de risque (/100)"].dtype.kind == "f"

    # Sans colonne de blessure : réponses par athlète quand même, sans vérité ni résultat.
    df = pd.read_csv(chemin, sep=";", decimal=",").drop(columns=["Injured"])
    df.to_csv(d / "sans_etiquette.csv", index=False)
    res2 = T.analyser(d / "sans_etiquette.csv", n_boot=20)
    assert "Résultat" not in res2["_lisible"].columns and "resultats" not in res2["par_athlete"]
    print(f"[OK] réponses par athlète : {pa['blessures_predites']}/400 prédits, comptes = précision/rappel du rapport, "
          "version groupée = version individuelle, CSV Excel relu")


def test_fiche_athlete():
    from analytics.fiche_athlete import construire_fiche, generer_pdf_fiche
    d = _tmp()
    res = T.analyser(T.jeu_de_repetition(d, n=300, graine=78), n_boot=20)
    L = res["_lisible"]
    i = int(L["Score de risque (/100)"].idxmax())
    f = construire_fiche(res["_X"][i], L.at[i, "Athlète"], blessure_reelle=L.at[i, "Blessure réelle"] == "Oui")
    assert f["profil_risque"]["injury_risk_pct"] == L.at[i, "Score de risque (/100)"]
    assert f["profil_risque"]["probabilite_prevalence_pct"] == L.at[i, "Probabilité estimée (%)"]
    assert f["verite"]["resultat"] == L.at[i, "Résultat"]
    assert f["plan_action"]["plan_combine"]["proba_avant_pct"] == f["profil_risque"]["probabilite_prevalence_pct"]
    assert "Régularité" not in f["performance"]["dimensions"]

    x = res["_X"][i].copy()
    x[FEATURE_NAMES.index("sommeil_h")] = np.nan
    f2 = construire_fiche(x, "Sommeil absent")
    assert any(c["champ"] == "sommeil_h" for c in f2["qualite"]["champs_imputes"]) and f2["verite"] is None
    assert generer_pdf_fiche(f).startswith(b"%PDF")
    print(f"[OK] fiche athlète : mêmes chiffres que le tableau ({f['profil_risque']['injury_risk_pct']} / 100), "
          "valeur absente signalée, PDF généré")


def test_zone_acwr_et_performance_sans_regularite():
    from analytics.metrics import calculer_performance_score, zone_acwr
    assert [zone_acwr(a)["zone"] for a in (0.7, 1.0, 1.4, 1.6)] == \
        ["Sous-charge", "Zone optimale", "Prudence", "Risque élevé"]
    bilan = {"tdee": 1.0, "kcal_consommees": 1.0, "renseigne": True}
    sans = calculer_performance_score({"acwr": 1.0}, 20, 3, bilan, None)
    avec = calculer_performance_score({"acwr": 1.0}, 20, 3, bilan, 70.0)
    assert "Régularité" not in sans["dimensions"] and len(avec["dimensions"]) == 5
    print("[OK] zones ACWR et score de performance sans régularité inventée")


if __name__ == "__main__":
    for nom, f in list(globals().items()):
        if nom.startswith("test_"):
            f()
    print("\nTous les tests de l'outil de test en direct passent.")
