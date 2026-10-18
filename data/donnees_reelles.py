# ATHÉNA - data/donnees_reelles.py
# Pont entre le MONDE RÉEL et les modèles : journal d'entraînement saisi par de vrais
# athlètes -> features du modèle -> score de risque -> évaluation HONNÊTE sur données réelles.
#
# Principe : on ne fabrique aucune donnée "réelle". Ce module (1) définit un journal
# quotidien simple (30 s de saisie par jour, faisable sur un téléphone), (2) valide les
# valeurs, (3) calcule les 10 features du modèle avec la méthode sRPE de Foster (2001),
# (4) mesure la performance du modèle de production sur les semaines où une blessure
# a (ou n'a pas) eu lieu — et refuse de conclure tant que l'échantillon est trop petit.
#
# Éthique : utiliser des identifiants pseudonymisés (A01, A02...) et obtenir le
# consentement écrit de chaque athlète (et d'un parent/tuteur si mineur).

from pathlib import Path

import numpy as np
import pandas as pd

DOSSIER_REEL = Path(__file__).parent / "real"
DOSSIER_REEL.mkdir(exist_ok=True)
CHEMIN_JOURNAL = DOSSIER_REEL / "journal.csv"

COLONNES = ["athlete_id", "date", "rpe", "duree_min", "sommeil_h", "fc_repos",
            "poids_kg", "douleur", "blessure"]

DESCRIPTION_COLONNES = {
    "athlete_id": "Identifiant pseudonymisé (ex. A01)",
    "date": "Jour, format AAAA-MM-JJ",
    "rpe": "Effort perçu de la séance, 0-10 (0 ou vide = repos)",
    "duree_min": "Durée de la séance en minutes (0 ou vide = repos)",
    "sommeil_h": "Heures dormies la nuit précédente",
    "fc_repos": "Fréquence cardiaque au réveil (bpm) — appli téléphone ou prise au poignet 30 s x 2",
    "poids_kg": "Poids du matin (optionnel)",
    "douleur": "Douleur/courbature ressentie, 0-10 (optionnel)",
    "blessure": "1 si blessure survenue ce jour-là (arrêt ou gêne obligeant à modifier l'entraînement), sinon 0",
}

# Plages plausibles : toute valeur hors plage est ignorée (comptée dans le rapport).
PLAGES = {
    "rpe": (0, 10), "duree_min": (0, 300), "sommeil_h": (2, 14), "fc_repos": (30, 120),
    "poids_kg": (30, 200), "douleur": (0, 10), "blessure": (0, 1),
}

# Valeurs de référence (médianes du jeu d'entraînement) pour ce que le journal ne mesure pas.
# Elles sont TOUJOURS signalées comme "non mesurées" dans le rapport de features.
VALEURS_REF = {
    "fc_repos_estimee": 62.0, "sommeil_h": 7.0, "age": 24.0, "imc": 22.5,
    "historique_blessures": 0.0, "deficit_calorique_pct": 0.0, "lactate": 2.2,
    "intensite_moyenne_7j": 0.55,
}

HISTORIQUE_MIN_JOURS = 14   # il faut au moins 2 semaines de journal pour calculer des features
HORIZON_JOURS = 7           # on prédit une blessure dans les 7 jours suivants


# ---------------------------------------------------------------- journal ----

def gabarit_csv() -> str:
    """CSV vide avec les bons en-têtes, à donner aux athlètes / entraîneurs."""
    return ",".join(COLONNES) + "\n"


def preparer_journal(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nettoie un journal brut. Ne lève pas d'exception sur des valeurs sales."""
    rapport = {"lignes_recues": int(len(df)), "valeurs_hors_plage": 0, "dates_invalides": 0,
               "doublons_supprimes": 0, "colonnes_manquantes": []}
    df = df.copy()
    for col in COLONNES:
        if col not in df.columns:
            rapport["colonnes_manquantes"].append(col)
            df[col] = np.nan
    df = df[COLONNES]

    df["athlete_id"] = df["athlete_id"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    rapport["dates_invalides"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])
    df = df[df["athlete_id"].ne("") & df["athlete_id"].ne("nan")]

    for col, (lo, hi) in PLAGES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        hors = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        rapport["valeurs_hors_plage"] += int(hors.sum())
        df.loc[hors, col] = np.nan

    avant = len(df)
    df = df.sort_values(["athlete_id", "date"]).drop_duplicates(["athlete_id", "date"], keep="last")
    rapport["doublons_supprimes"] = int(avant - len(df))
    rapport["lignes_valides"] = int(len(df))
    return df.reset_index(drop=True), rapport


def charger_journal() -> pd.DataFrame:
    if not CHEMIN_JOURNAL.exists():
        return pd.DataFrame(columns=COLONNES)
    df, _ = preparer_journal(pd.read_csv(CHEMIN_JOURNAL))
    return df


def ajouter_entree(entree: dict) -> pd.DataFrame:
    """Ajoute (ou remplace) une ligne du journal et sauvegarde sur disque."""
    actuel = charger_journal()
    nouvelle = pd.DataFrame([{c: entree.get(c, np.nan) for c in COLONNES}])
    fusion, _ = preparer_journal(pd.concat([actuel, nouvelle], ignore_index=True))
    fusion.to_csv(CHEMIN_JOURNAL, index=False)
    return fusion


def importer_csv(contenu) -> tuple[pd.DataFrame, dict]:
    """Fusionne un CSV (chemin ou fichier uploadé) au journal existant."""
    brut = pd.read_csv(contenu)
    nouveau, rapport = preparer_journal(brut)
    fusion, _ = preparer_journal(pd.concat([charger_journal(), nouveau], ignore_index=True))
    fusion.to_csv(CHEMIN_JOURNAL, index=False)
    rapport["lignes_totales_journal"] = int(len(fusion))
    return fusion, rapport


def adapter_csv_externe(chemin, correspondance: dict, **kw) -> tuple[pd.DataFrame, dict]:
    """Convertit un jeu de données PUBLIC (autre format) vers le format ATHÉNA.

    `correspondance` : {colonne_athena: colonne_du_fichier_source}. Exemple :
        {"athlete_id": "Athlete ID", "date": "Date", "rpe": "perceived exertion",
         "duree_min": "total minutes", "sommeil_h": "sleep hours", "blessure": "injury"}
    Les colonnes non fournies restent vides (le modèle utilise alors des valeurs de
    référence, signalées comme telles).
    """
    brut = pd.read_csv(chemin, **kw)
    renomme = pd.DataFrame({k: brut[v] for k, v in correspondance.items() if v in brut.columns})
    return preparer_journal(renomme)


# --------------------------------------------------------------- features ----

def serie_quotidienne(df_athlete: pd.DataFrame) -> pd.DataFrame:
    """Série jour par jour. Charge = RPE x durée (sRPE, Foster 2001).
    Un jour sans ligne est compté comme repos (charge 0) ; `taux_saisie` mesure l'assiduité."""
    d = df_athlete.set_index("date").sort_index()
    jours = pd.date_range(d.index.min(), d.index.max(), freq="D")
    s = d.reindex(jours)
    s["saisi"] = s["athlete_id"].notna()
    s["charge"] = (s["rpe"].fillna(0) * s["duree_min"].fillna(0)).astype(float)
    s["blessure"] = s["blessure"].fillna(0)
    return s


def features_a_date(serie: pd.DataFrame, t: pd.Timestamp, profil: dict | None = None) -> tuple[np.ndarray, dict]:
    """Calcule les 10 features du modèle à la date t, avec les données jusqu'à t inclus."""
    profil = profil or {}
    hist = serie.loc[:t]
    aigue = hist["charge"].tail(7).mean()
    chronique = hist["charge"].tail(28).mean()
    acwr = float(aigue / chronique) if chronique > 0 else np.nan

    semaine = hist.tail(7)
    sessions = semaine[semaine["charge"] > 0]
    mesures, ref = {}, []

    def _val(nom_feature, valeur, cle_ref=None):
        if valeur is None or not np.isfinite(valeur):
            mesures[nom_feature] = VALEURS_REF[cle_ref or nom_feature]
            ref.append(nom_feature)
        else:
            mesures[nom_feature] = float(valeur)

    fc = semaine["fc_repos"].mean()
    sommeil = semaine["sommeil_h"].mean()
    intensite = sessions["rpe"].mean() / 10 if len(sessions) else np.nan

    taille = profil.get("taille_cm")
    poids = hist["poids_kg"].dropna().tail(7).mean() if hist["poids_kg"].notna().any() else np.nan
    imc = poids / ((taille / 100) ** 2) if taille and np.isfinite(poids) else np.nan

    if "historique_blessures" in profil:
        hist_bl = profil["historique_blessures"]
    else:
        jours_bl = hist.index[hist["blessure"] == 1]
        an = jours_bl[jours_bl >= t - pd.Timedelta(days=365)]
        hist_bl = 0 if len(an) == 0 else 1 + int((pd.Series(an).diff().dt.days.dropna() > 7).sum())

    if np.isfinite(acwr):
        mesures["acwr"] = acwr
    else:                       # aucune charge chronique : ACWR neutre (1.0), signalé comme non mesuré
        mesures["acwr"] = 1.0
        ref.append("acwr")
    mesures["charge_chronique"] = float(chronique) if np.isfinite(chronique) else 0.0
    _val("fc_repos_estimee", fc)
    _val("sommeil_h", sommeil)
    _val("age", profil.get("age"))
    _val("imc", imc)
    _val("historique_blessures", hist_bl)
    _val("deficit_calorique_pct", None)          # non mesuré par le journal
    _val("lactate", None)                        # non mesuré par le journal
    _val("intensite_moyenne_7j", intensite)

    from models.train_injury_model import FEATURE_NAMES
    vecteur = np.array([[mesures[n] for n in FEATURE_NAMES]], dtype=float)
    return vecteur, {"non_mesures": ref, "jours_historique": int(len(hist)),
                     "taux_saisie_28j": round(float(hist["saisi"].tail(28).mean()), 2)}


def risque_actuel(df_journal: pd.DataFrame, athlete_id: str, profil: dict | None = None) -> dict | None:
    """Score de risque d'un athlète réel à partir de son journal (None si < 14 jours)."""
    d = df_journal[df_journal["athlete_id"] == athlete_id]
    if d.empty:
        return None
    serie = serie_quotidienne(d)
    if len(serie) < HISTORIQUE_MIN_JOURS:
        return {"statut": "historique_insuffisant", "jours": int(len(serie)),
                "jours_requis": HISTORIQUE_MIN_JOURS}
    from models.injury_risk import predire_profil_risque
    vecteur, info = features_a_date(serie, serie.index[-1], profil)
    return {"statut": "ok", "profil": predire_profil_risque(vecteur), "vecteur": vecteur,
            "info": info, "serie": serie}


# ------------------------------------------------------------- évaluation ----

def construire_echantillons(df_journal: pd.DataFrame, profils: dict | None = None) -> pd.DataFrame:
    """Un échantillon = (athlète, jour t) avec ≥14 jours d'historique et 7 jours de suivi complets.
    Étiquette = blessure survenue dans les 7 jours qui suivent t. Les jours où l'athlète est
    déjà blessé (blessure le jour t) sont exclus."""
    profils = profils or {}
    lignes = []
    from models.injury_risk import predire_profil_risque
    for athlete, d in df_journal.groupby("athlete_id"):
        serie = serie_quotidienne(d)
        if len(serie) < HISTORIQUE_MIN_JOURS + HORIZON_JOURS:
            continue
        for i in range(HISTORIQUE_MIN_JOURS - 1, len(serie) - HORIZON_JOURS):
            t = serie.index[i]
            if serie["blessure"].iloc[i] == 1:
                continue
            futur = serie["blessure"].iloc[i + 1: i + 1 + HORIZON_JOURS]
            vecteur, _ = features_a_date(serie, t, profils.get(athlete))
            p = predire_profil_risque(vecteur)
            lignes.append({"athlete_id": athlete, "date": t, "y": int(futur.max() >= 1),
                           "score_modele": p["injury_risk_pct"], "acwr": float(vecteur[0][0]),
                           "sommeil_h": float(vecteur[0][3])})
    return pd.DataFrame(lignes)


def taux_saisie_global(df_journal: pd.DataFrame) -> float:
    """Part des jours effectivement renseignés, entre le premier et le dernier jour de chaque athlète.
    Proche de 1 = données quotidiennes. Faible = données hebdomadaires ou très lacunaires : les jours
    manquants seraient alors comptés à tort comme des jours de repos."""
    if df_journal.empty:
        return 0.0
    g = df_journal.groupby("athlete_id")["date"].agg(["count", "min", "max"])
    jours = (g["max"] - g["min"]).dt.days + 1
    return float(g["count"].sum() / jours.sum())


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def evaluer_sur_donnees_reelles(df_journal: pd.DataFrame, profils: dict | None = None,
                                n_bootstrap: int = 1000, graine: int = 42,
                                min_athletes: int = 5, min_positifs: int = 15) -> dict:
    """AUC du modèle de production sur données réelles, avec IC95 par bootstrap PAR ATHLÈTE
    (les jours d'un même athlète sont corrélés : rééchantillonner les jours gonflerait la
    confiance). Compare à deux baselines naïves : ACWR seul, et sommeil seul (inversé).
    Retourne statut='insuffisant' tant que l'échantillon ne permet pas de conclure."""
    taux = taux_saisie_global(df_journal)
    if taux < 0.6:
        return {"statut": "resolution_inadaptee", "taux_saisie": round(taux, 2),
                "message": (f"Seulement {taux:.0%} des jours sont renseignés : les jours manquants seraient comptés "
                            "comme des jours de repos et fausseraient charge et ACWR. Cette évaluation exige des "
                            "données quotidiennes (≥ 60 % des jours renseignés). Un jeu de données hebdomadaire "
                            "demande un adaptateur dédié.")}
    ech = construire_echantillons(df_journal, profils)
    base = {"n_echantillons": int(len(ech)), "n_athletes": int(ech["athlete_id"].nunique()) if len(ech) else 0,
            "n_positifs": int(ech["y"].sum()) if len(ech) else 0}
    base["n_negatifs"] = base["n_echantillons"] - base["n_positifs"]

    if base["n_athletes"] < min_athletes or base["n_positifs"] < min_positifs or base["n_negatifs"] < min_positifs:
        base["statut"] = "insuffisant"
        base["message"] = (
            f"Échantillon trop petit pour conclure : il faut au moins {min_athletes} athlètes et "
            f"{min_positifs} fenêtres avec blessure ET sans blessure "
            f"(actuellement {base['n_athletes']} athlètes, {base['n_positifs']} avec blessure, "
            f"{base['n_negatifs']} sans). Continuer la collecte.")
        return base

    y, s = ech["y"].values, ech["score_modele"].values
    base["auc_modele"] = round(_auc(y, s), 3)
    base["auc_acwr_seul"] = round(_auc(y, ech["acwr"].values), 3)
    base["auc_sommeil_seul"] = round(_auc(y, -ech["sommeil_h"].values), 3)

    rng = np.random.default_rng(graine)
    athletes = ech["athlete_id"].unique()
    groupes = {a: ech[ech["athlete_id"] == a] for a in athletes}
    aucs = []
    for _ in range(n_bootstrap):
        tir = rng.choice(athletes, size=len(athletes), replace=True)
        b = pd.concat([groupes[a] for a in tir])
        if b["y"].nunique() < 2:
            continue
        aucs.append(_auc(b["y"].values, b["score_modele"].values))
    base["ic95_auc"] = [round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)]
    base["statut"] = "ok"
    base["lecture"] = ("AUC 0.5 = hasard. Si l'IC95 inclut 0.5, le modèle n'est PAS démontré meilleur "
                       "que le hasard sur ces données réelles — c'est un résultat à rapporter tel quel.")
    return base
