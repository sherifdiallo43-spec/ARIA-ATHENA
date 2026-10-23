import numpy as np

from .injury_risk import predire_profil_risque, construire_vecteur_features


def _verifier_injury_risk_valide(profil: dict) -> None:
    pct = profil.get("injury_risk_pct")
    assert isinstance(pct, (int, float)), f"injury_risk_pct non numérique : {pct!r}"
    assert 0.0 <= float(pct) <= 100.0, f"injury_risk_pct hors [0, 100] : {pct!r}"
    assert "qualite_donnees" in profil, "clé 'qualite_donnees' manquante"


def test_vecteur_valide():
    vec = construire_vecteur_features(
        acwr=1.2, charge_chronique=450, fc_repos=60, sommeil_h=7.5,
        age=25, imc=22.0, historique_blessures=1, deficit_calorique_pct=-5.0,
        lactate=4.0, intensite_moyenne_7j=0.6,
    )
    profil = predire_profil_risque(vec)
    _verifier_injury_risk_valide(profil)
    q = profil["qualite_donnees"]
    assert q["champs_imputes"] == [], f"imputations inattendues : {q['champs_imputes']}"
    assert q["champs_clippes"] == [], f"clippages inattendus : {q['champs_clippes']}"
    print(f"[OK] Cas 1 — vecteur valide, injury_risk_pct={profil['injury_risk_pct']}%")


def test_nan_sommeil():
    vec = construire_vecteur_features(
        acwr=1.2, charge_chronique=450, fc_repos=60, sommeil_h=np.nan,
        age=25, imc=22.0, historique_blessures=1, deficit_calorique_pct=-5.0,
        lactate=4.0, intensite_moyenne_7j=0.6,
    )
    profil = predire_profil_risque(vec)
    _verifier_injury_risk_valide(profil)
    q = profil["qualite_donnees"]
    champs = [c["champ"] for c in q["champs_imputes"]]
    assert "sommeil_h" in champs, f"sommeil_h non imputé : {q}"
    print(f"[OK] Cas 2 — NaN sommeil_h imputé, injury_risk_pct={profil['injury_risk_pct']}%")


def test_age_aberrant():
    vec = construire_vecteur_features(
        acwr=1.2, charge_chronique=450, fc_repos=60, sommeil_h=7.5,
        age=200, imc=22.0, historique_blessures=1, deficit_calorique_pct=-5.0,
        lactate=4.0, intensite_moyenne_7j=0.6,
    )
    profil = predire_profil_risque(vec)
    _verifier_injury_risk_valide(profil)
    q = profil["qualite_donnees"]
    clippes = {c["champ"]: c for c in q["champs_clippes"]}
    assert "age" in clippes, f"age non clippé : {q}"
    assert clippes["age"]["valeur_utilisee"] == 40.0, f"borne haute age attendue : {clippes['age']}"
    print(f"[OK] Cas 3 — age=200 clippé à 40, injury_risk_pct={profil['injury_risk_pct']}%")


def test_plusieurs_none():
    vec = construire_vecteur_features(
        acwr=None, charge_chronique=None, fc_repos=60, sommeil_h=7.5,
        age=None, imc=22.0, historique_blessures=1, deficit_calorique_pct=-5.0,
        lactate=4.0, intensite_moyenne_7j=0.6,
    )
    profil = predire_profil_risque(vec)
    _verifier_injury_risk_valide(profil)
    q = profil["qualite_donnees"]
    champs = {c["champ"] for c in q["champs_imputes"]}
    assert {"acwr", "charge_chronique", "age"}.issubset(champs), \
        f"champs attendus manquants dans les imputations : {champs}"
    print(f"[OK] Cas 4 — {len(q['champs_imputes'])} imputations, "
          f"injury_risk_pct={profil['injury_risk_pct']}%")


def test_types_inattendus():
    vec = construire_vecteur_features(
        acwr="pas un nombre", charge_chronique="???", fc_repos=60, sommeil_h=7.5,
        age=[1, 2, 3], imc=22.0, historique_blessures=1, deficit_calorique_pct=-5.0,
        lactate=4.0, intensite_moyenne_7j=0.6,
    )
    profil = predire_profil_risque(vec)
    _verifier_injury_risk_valide(profil)
    print(f"[OK] Cas bonus — types exotiques gérés, injury_risk_pct={profil['injury_risk_pct']}%")


def test_signe_deficit():
    # Convention du modèle : déficit POSITIF = l'athlète mange moins que son besoin -> risque plus élevé.
    def risque(deficit):
        return predire_profil_risque(construire_vecteur_features(
            acwr=1.1, charge_chronique=420, fc_repos=62, sommeil_h=7.0, age=23, imc=22.5,
            historique_blessures=0, deficit_calorique_pct=deficit, lactate=2.2,
            intensite_moyenne_7j=0.55))["injury_risk_pct"]
    r_deficit, r_neutre, r_surplus = risque(30), risque(0), risque(-30)
    assert r_deficit > r_neutre >= r_surplus, (r_deficit, r_neutre, r_surplus)
    print(f"[OK] Signe du déficit — manger 30 % de moins : {r_neutre}% -> {r_deficit}%")


def test_imputation_valeurs_typiques():
    # Une variable absente reçoit sa valeur TYPIQUE (ex. 0 blessure passée), pas le milieu du domaine (5).
    from .train_injury_model import VALEURS_TYPIQUES
    vec = construire_vecteur_features(
        acwr=None, charge_chronique=420, fc_repos=62, sommeil_h=7.0, age=23, imc=22.5,
        historique_blessures=None, deficit_calorique_pct=0, lactate=None, intensite_moyenne_7j=0.55,
    )
    q = predire_profil_risque(vec)["qualite_donnees"]
    imputees = {c["champ"]: c["valeur_imputee"] for c in q["champs_imputes"]}
    attendu = {k: VALEURS_TYPIQUES[k] for k in ("acwr", "historique_blessures", "lactate")}
    assert imputees == attendu, imputees
    print(f"[OK] Imputation par valeurs typiques : {imputees}")


def lancer_tous_les_tests():
    print("=== Tests de robustesse ATHÉNA — données manquantes / aberrantes ===\n")
    test_vecteur_valide()
    test_nan_sommeil()
    test_age_aberrant()
    test_plusieurs_none()
    test_types_inattendus()
    test_signe_deficit()
    test_imputation_valeurs_typiques()
    print("\n=== Tous les tests sont passés sans exception. ===")


if __name__ == "__main__":
    lancer_tous_les_tests()
