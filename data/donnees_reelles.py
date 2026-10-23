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

import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

DOSSIER_REEL = Path(__file__).parent / "real"
DOSSIER_REEL.mkdir(exist_ok=True)
CHEMIN_JOURNAL = DOSSIER_REEL / "journal.csv"
CHEMIN_REPAS = DOSSIER_REEL / "repas.csv"

COLONNES = ["athlete_id", "date", "rpe", "duree_min", "sommeil_h", "fc_repos",
            "poids_kg", "douleur", "blessure"]
COLONNES_REPAS = ["id", "athlete_id", "date", "heure", "repas", "kcal",
                  "proteines_g", "glucides_g", "lipides_g"]


class ErreurJournal(Exception):
    """Erreur à montrer telle quelle à l'utilisateur (message en français, sans trace Python)."""

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
    # 600 min : la durée du jour additionne les séances (ex. 2 x 3 h).
    "rpe": (0, 10), "duree_min": (0, 600), "sommeil_h": (2, 14), "fc_repos": (30, 120),
    "poids_kg": (30, 200), "douleur": (0, 10), "blessure": (0, 1),
}

# Valeurs de référence pour ce que le journal ne mesure pas : la même table que le dashboard et le
# testeur CSV (valeurs typiques du jeu d'entraînement). Elles sont TOUJOURS signalées comme
# "non mesurées" dans le rapport de features.
from models.train_injury_model import VALEURS_TYPIQUES as VALEURS_REF  # noqa: E402

HISTORIQUE_MIN_JOURS = 14   # il faut au moins 2 semaines de journal pour calculer des features
HORIZON_JOURS = 7           # on prédit une blessure dans les 7 jours suivants


# ---------------------------------------------------------------- journal ----

def gabarit_csv() -> str:
    """CSV vide avec les bons en-têtes, à donner aux athlètes / entraîneurs."""
    return ",".join(COLONNES) + "\n"


def normaliser_id(ids) -> pd.Series:
    """Identifiants comparables : texte, sans espaces autour, en majuscules (« a01 » = « A01 »).
    Un identifiant absent devient "" (et la ligne est ensuite écartée)."""
    s = pd.Series(ids, dtype="string").str.strip().str.upper().fillna("")
    return s.where(~s.isin(["NAN", "NONE", "<NA>"]), "").astype(object)


def lire_dates(valeurs) -> pd.Series:
    """Dates AAAA-MM-JJ, puis à défaut au format français JJ/MM/AAAA (jamais MM/JJ)."""
    texte = pd.Series(valeurs).astype("string").str.strip()
    dates = pd.to_datetime(texte, errors="coerce", format="ISO8601")
    reste = dates.isna() & texte.notna()
    if reste.any():
        dates[reste] = pd.to_datetime(texte[reste], errors="coerce", dayfirst=True, format="mixed")
    return dates.dt.normalize()


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

    df["athlete_id"] = normaliser_id(df["athlete_id"])
    df["date"] = lire_dates(df["date"])
    rapport["dates_invalides"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])
    df = df[df["athlete_id"].ne("")]

    for col, (lo, hi) in PLAGES.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Export Excel français : « 7,5 » -> 7.5 ; « oui »/« non » acceptés pour la blessure.
            texte = df[col].astype(str).str.strip().str.replace(",", ".", regex=False).str.lower()
            if col == "blessure":
                texte = texte.replace({"oui": "1", "non": "0", "yes": "1", "no": "0", "true": "1", "false": "0"})
            df[col] = texte
        df[col] = pd.to_numeric(df[col], errors="coerce")
        hors = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        rapport["valeurs_hors_plage"] += int(hors.sum())
        df.loc[hors, col] = np.nan

    avant = len(df)
    df = df.sort_values(["athlete_id", "date"]).drop_duplicates(["athlete_id", "date"], keep="last")
    rapport["doublons_supprimes"] = int(avant - len(df))
    rapport["lignes_valides"] = int(len(df))
    return df.reset_index(drop=True), rapport


def _ecrire_atomique(df: pd.DataFrame, chemin: Path) -> None:
    """Écrit dans un fichier temporaire puis le renomme : une coupure ne laisse jamais
    un fichier à moitié écrit. UTF-8 avec BOM pour qu'Excel affiche bien les accents."""
    tmp = chemin.with_name(chemin.name + ".tmp")
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        os.replace(tmp, chemin)
    except PermissionError:
        raise ErreurJournal(f"Impossible d'enregistrer : le fichier {chemin.name} est ouvert dans un autre "
                            "programme (Excel ?). Ferme-le puis recommence.") from None
    finally:
        if tmp.exists():
            tmp.unlink()


def _lire_fichier(chemin: Path) -> pd.DataFrame | None:
    """Contenu brut d'un fichier de données, None s'il n'existe pas ou est vide."""
    if not chemin.exists() or chemin.stat().st_size == 0:
        return None
    try:
        return lire_csv_robuste(chemin, dtype={"athlete_id": str, "id": str})
    except (ValueError, PermissionError) as e:
        raise ErreurJournal(f"Le fichier {chemin.name} est illisible ({e}). Il n'a pas été modifié : "
                            "corrige-le ou restaure la dernière sauvegarde exportée.") from None


def charger_journal() -> pd.DataFrame:
    brut = _lire_fichier(CHEMIN_JOURNAL)
    if brut is None:
        return pd.DataFrame(columns=COLONNES)
    df, _ = preparer_journal(reconnaitre_colonnes(brut)[0])
    if len(brut) and df.empty:
        # Sans ce garde-fou, la sauvegarde suivante réécrirait le journal avec une seule ligne.
        raise ErreurJournal(f"{CHEMIN_JOURNAL.name} contient {len(brut)} ligne(s) mais aucune n'est lisible "
                            "(identifiants ou dates). Le journal n'a pas été modifié.")
    return df


def ajouter_entree(entree: dict) -> tuple[pd.DataFrame, str]:
    """Ajoute (ou remplace) la journée d'un athlète et sauvegarde sur disque.
    Retourne (journal, statut) avec statut "ajoutee" ou "remplacee"."""
    athlete = normaliser_id([entree.get("athlete_id")]).iloc[0]
    if not athlete:
        raise ErreurJournal("Identifiant vide : indique l'identifiant de l'athlète (ex. A01).")
    date = lire_dates([entree.get("date")]).iloc[0]
    if pd.isna(date):
        raise ErreurJournal("Date illisible : utilise le format AAAA-MM-JJ ou JJ/MM/AAAA.")
    actuel = charger_journal()
    existait = bool(((actuel["athlete_id"] == athlete) & (actuel["date"] == date)).any())
    nouvelle = pd.DataFrame([{c: entree.get(c, np.nan) for c in COLONNES}])
    nouvelle["athlete_id"], nouvelle["date"] = athlete, date
    fusion, _ = preparer_journal(pd.concat([actuel, nouvelle], ignore_index=True))
    _ecrire_atomique(fusion, CHEMIN_JOURNAL)
    return fusion, "remplacee" if existait else "ajoutee"


def ajouter_seance(athlete_id, date, rpe: float, duree_min: float) -> tuple[pd.DataFrame, dict]:
    """Ajoute une séance à la journée d'un athlète SANS effacer le reste de la journée
    (sommeil, FC...). Deux séances le même jour sont cumulées : durées additionnées et
    RPE pondéré pour que RPE x durée = somme des charges (sRPE de Foster)."""
    athlete = normaliser_id([athlete_id]).iloc[0]
    if not athlete:
        raise ErreurJournal("Identifiant vide : indique l'identifiant de l'athlète (ex. A01).")
    if duree_min <= 0:
        raise ErreurJournal("Durée nulle : indique la durée de la séance en minutes.")
    jour = lire_dates([date]).iloc[0]
    actuel = charger_journal()
    masque = (actuel["athlete_id"] == athlete) & (actuel["date"] == jour)
    entree = actuel[masque].iloc[0].to_dict() if masque.any() else {"athlete_id": athlete, "date": jour}
    charge_avant = float(np.nan_to_num(entree.get("rpe", np.nan)) * np.nan_to_num(entree.get("duree_min", np.nan)))
    duree_totale = float(np.nan_to_num(entree.get("duree_min", np.nan))) + float(duree_min)
    charge_totale = charge_avant + float(rpe) * float(duree_min)
    entree["duree_min"] = duree_totale
    entree["rpe"] = round(charge_totale / duree_totale, 2)
    journal, _ = ajouter_entree(entree)
    return journal, {"cumulee": bool(masque.any()), "charge_jour": round(charge_totale),
                     "duree_jour": round(duree_totale)}


def charge_28j(df_journal: pd.DataFrame, athlete_id: str, fin=None) -> pd.DataFrame:
    """Charge jour par jour (RPE x durée) d'un athlète réel, jusqu'à `fin` (aujourd'hui par
    défaut), au format de `calculer_acwr` : colonnes date, charge. Jours sans saisie = 0."""
    fin = pd.Timestamp(fin if fin is not None else pd.Timestamp.today()).normalize()
    d = df_journal[(df_journal["athlete_id"] == athlete_id) & (df_journal["date"] <= fin)]
    if d.empty:
        return pd.DataFrame(columns=["date", "charge"])
    serie = serie_quotidienne(d)
    debut = max(serie.index.min(), fin - pd.Timedelta(days=27))
    jours = pd.date_range(debut, fin, freq="D")
    charge = serie["charge"].reindex(jours, fill_value=0.0)
    return pd.DataFrame({"date": jours.date, "charge": charge.values})


def moyennes_7j(df_journal: pd.DataFrame, athlete_id: str, fin=None) -> dict:
    """Moyennes du sommeil et de la FC au réveil sur les 7 derniers jours (NaN si non saisis)."""
    fin = pd.Timestamp(fin if fin is not None else pd.Timestamp.today()).normalize()
    d = df_journal[(df_journal["athlete_id"] == athlete_id)
                   & (df_journal["date"] > fin - pd.Timedelta(days=7)) & (df_journal["date"] <= fin)]
    return {"sommeil_h": float(d["sommeil_h"].mean()) if len(d) else np.nan,
            "fc_repos": float(d["fc_repos"].mean()) if len(d) else np.nan}


# ------------------------------------------------------------------ repas ----

def charger_repas(athlete_id=None, jour=None) -> pd.DataFrame:
    """Repas enregistrés, filtrés par athlète et/ou par jour si demandé."""
    brut = _lire_fichier(CHEMIN_REPAS)
    if brut is None:
        return pd.DataFrame(columns=COLONNES_REPAS)
    df = brut.reindex(columns=COLONNES_REPAS)
    df["id"] = df["id"].astype(str)
    df["athlete_id"] = normaliser_id(df["athlete_id"])
    df["date"] = lire_dates(df["date"])
    for col in ("kcal", "proteines_g", "glucides_g", "lipides_g"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df.dropna(subset=["date"])
    if athlete_id is not None:
        df = df[df["athlete_id"] == normaliser_id([athlete_id]).iloc[0]]
    if jour is not None:
        df = df[df["date"] == lire_dates([jour]).iloc[0]]
    return df.sort_values(["date", "heure"]).reset_index(drop=True)


def ajouter_repas(athlete_id, jour, heure: str, repas: str, kcal: float,
                  proteines_g: float = 0, glucides_g: float = 0, lipides_g: float = 0) -> dict:
    """Enregistre un repas sur disque. Retourne la ligne enregistrée."""
    athlete = normaliser_id([athlete_id]).iloc[0]
    if not athlete:
        raise ErreurJournal("Aucun athlète sélectionné pour ce repas.")
    if not kcal or kcal <= 0:
        raise ErreurJournal("Indique l'apport du repas en kcal (supérieur à 0).")
    ligne = {"id": uuid.uuid4().hex[:8], "athlete_id": athlete, "date": lire_dates([jour]).iloc[0],
             "heure": heure, "repas": (repas or "").strip() or f"Repas de {heure}", "kcal": kcal,
             "proteines_g": proteines_g, "glucides_g": glucides_g, "lipides_g": lipides_g}
    tous = charger_repas()
    tous = pd.concat([tous, pd.DataFrame([ligne])], ignore_index=True) if len(tous) else pd.DataFrame([ligne])
    _ecrire_atomique(tous[COLONNES_REPAS], CHEMIN_REPAS)
    return ligne


def supprimer_repas(id_repas: str) -> None:
    tous = charger_repas()
    _ecrire_atomique(tous[tous["id"] != str(id_repas)][COLONNES_REPAS], CHEMIN_REPAS)


# Noms acceptés pour chaque colonne du journal (normalisés : minuscules, sans accents,
# séparateurs -> « _ »), pour lire un export Excel ou un fichier en anglais sans retouche.
SYNONYMES_JOURNAL = {
    "athlete_id": ["athlete_id", "id_athlete", "athlete", "id", "joueur", "player", "player_id", "code_athlete"],
    "date": ["date", "jour", "day", "date_seance", "session_date"],
    "rpe": ["rpe", "effort", "effort_percu", "perceived_exertion", "rpe_0_10"],
    "duree_min": ["duree_min", "duree", "duree_seance", "duration", "duration_min", "minutes", "temps_min"],
    "sommeil_h": ["sommeil_h", "sommeil", "heures_sommeil", "sleep", "sleep_h", "sleep_hours"],
    "fc_repos": ["fc_repos", "fc_reveil", "fc_au_repos", "resting_hr", "resting_heart_rate", "rhr"],
    "poids_kg": ["poids_kg", "poids", "weight", "weight_kg", "masse"],
    "douleur": ["douleur", "courbatures", "pain", "soreness"],
    "blessure": ["blessure", "blesse", "injury", "injured", "is_injured"],
}
COLONNES_OBLIGATOIRES = ["athlete_id", "date"]


def _norm_nom(s) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def lire_csv_robuste(contenu, **kw) -> pd.DataFrame:
    """Lit un CSV quel que soit le séparateur (, ; tabulation) et l'encodage (UTF-8, Excel/Latin-1)."""
    derniere = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            if hasattr(contenu, "seek"):
                contenu.seek(0)
            df = pd.read_csv(contenu, sep=None, engine="python", encoding=enc, **kw)
            if df.shape[1] >= 2:
                return df
        except Exception as e:  # noqa: BLE001 - on tente l'encodage suivant
            derniere = e
    raise ValueError(f"Lecture du fichier impossible : ce n'est pas un CSV lisible ({derniere}).")


def reconnaitre_colonnes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Renomme les colonnes reconnues vers le format du journal. Retourne (df, {colonne_journal: colonne_source})."""
    normalisees = {_norm_nom(c): c for c in df.columns}
    trouvees = {}
    for cible, synonymes in SYNONYMES_JOURNAL.items():
        for syn in synonymes:
            if syn in normalisees and normalisees[syn] not in trouvees.values():
                trouvees[cible] = normalisees[syn]
                break
    return df.rename(columns={src: cible for cible, src in trouvees.items()}), trouvees


def importer_csv(contenu) -> tuple[pd.DataFrame, dict]:
    """Fusionne un CSV (chemin ou fichier uploadé) au journal existant.

    Ne modifie JAMAIS le journal si le fichier est incompatible ou ne contient aucune ligne
    valide. `rapport["statut"]` vaut "ok", "incompatible" ou "vide", avec un message clair."""
    actuel = charger_journal()
    # Tout en texte : un identifiant « 007 » reste « 007 » (preparer_journal convertit les nombres).
    brut = lire_csv_robuste(contenu, dtype=str)
    renomme, trouvees = reconnaitre_colonnes(brut)
    rapport = {"colonnes_fichier": [str(c) for c in brut.columns], "colonnes_reconnues": trouvees,
               "lignes_recues": int(len(brut))}

    manquantes = [c for c in COLONNES_OBLIGATOIRES if c not in trouvees]
    if manquantes:
        rapport.update(statut="incompatible", lignes_valides=0, lignes_totales_journal=int(len(actuel)),
                       message=("Ce fichier n'est pas un journal d'entraînement : colonne(s) obligatoire(s) "
                                f"introuvable(s) : {', '.join(manquantes)}. Un journal donne, pour chaque athlète "
                                "et chaque jour, l'effort, la durée, le sommeil et les blessures (voir le gabarit). "
                                "Pour tester le modèle sur un jeu de données, utiliser l'onglet « Test CSV »."))
        return actuel, rapport

    nouveau, rapport_prep = preparer_journal(renomme)
    rapport.update(rapport_prep)
    if rapport["lignes_valides"] == 0:
        rapport.update(statut="vide", lignes_totales_journal=int(len(actuel)),
                       message=(f"Aucune ligne valide sur {rapport['lignes_recues']} : "
                                f"{rapport['dates_invalides']} date(s) illisible(s) (format attendu AAAA-MM-JJ). "
                                "Le journal n'a pas été modifié."))
        return actuel, rapport

    fusion, _ = preparer_journal(pd.concat([actuel, nouveau], ignore_index=True))
    _ecrire_atomique(fusion, CHEMIN_JOURNAL)
    rapport["lignes_totales_journal"] = int(len(fusion))
    rapport["statut"] = "ok"
    sans_charge = [c for c in ("rpe", "duree_min") if c not in trouvees]
    rapport["avertissements"] = []
    if sans_charge:
        rapport["avertissements"].append("Colonne(s) absente(s) : " + ", ".join(sans_charge)
                                         + " : la charge d'entraînement (RPE × durée) ne peut pas être calculée.")
    if "blessure" not in trouvees:
        rapport["avertissements"].append("Pas de colonne « blessure » : le risque sera calculé, mais la "
                                         "performance du modèle ne pourra pas être évaluée sur ces données.")
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
