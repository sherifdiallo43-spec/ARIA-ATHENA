# ATHÉNA - models/testeur_csv.py
# Test en direct du modèle sur un CSV INÉDIT (ex. jeu fourni par le jury le jour de la finale).
#
# Ce que fait l'outil, sans rien deviner en silence :
#   1. lit le CSV (séparateur , ou ; détecté, virgule décimale acceptée, BOM/UTF-8 ou Latin-1) ;
#   2. associe les colonnes du fichier aux 10 variables du modèle (noms exacts, synonymes FR/EN,
#      ou fichier --mapping) et AFFICHE l'association, y compris les variables absentes ;
#   3. détecte les unités courantes (sommeil en minutes, intensité sur 10 ou 100, bilan calorique au lieu
#      d'un déficit) et les convertit en le disant, puis nettoie comme en production (valeur manquante ->
#      valeur typique de l'entraînement, valeur hors domaine -> bornée) et compte chaque correction ;
#   4. prédit : score de risque, probabilité corrigée de la prévalence, fatigue ;
#   5. si une colonne d'étiquette existe : ROC-AUC avec IC95 (bootstrap), précision/rappel, rappel parmi les
#      20 % les plus à risque, Brier, fiabilité par tranches ; si une colonne fatigue existe : MAE/RMSE/R² ;
#   6. option --train : réentraîne les modèles de l'étude (GB d = 1, 2, 3, régression logistique) sur un
#      CSV d'entraînement fourni et les compare sur le CSV de test, avec le modèle de production.
#
# Usage :
#   python -m models.testeur_csv --test inedit.csv
#   python -m models.testeur_csv --test inedit.csv --cible blessure --mapping mon_mapping.json
#   python -m models.testeur_csv --train train.csv --test inedit.csv
#   python -m models.testeur_csv --demo        # répétition générale : jeu inédit simulé, colonnes renommées
#
# Sorties (dossier --sortie, défaut sorties_test/) : predictions.csv, rapport.md, resume.json.

import argparse
import difflib
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

if __package__ in (None, ""):                       # lancement direct : python models/testeur_csv.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "models"

from .train_injury_model import FEATURE_NAMES, VALEURS_TYPIQUES, _generer_dataset  # noqa: E402
from .injury_risk import (_charger_modeles, corriger_prevalence, DOMAINE_ENTRAINEMENT, LABELS_LISIBLES)  # noqa: E402

# Synonymes (déjà normalisés : minuscules, sans accents, séparateurs -> « _ »).
SYNONYMES = {
    "acwr": ["acwr", "ratio_charge", "acute_chronic_workload_ratio", "workload_ratio", "ratio_acwr", "rapport_charge"],
    "charge_chronique": ["charge_chronique", "chronic_load", "chronic_workload", "charge_28j", "load_28d", "charge_28"],
    "fc_repos_estimee": ["fc_repos_estimee", "fc_repos", "fc_au_repos", "resting_hr", "resting_heart_rate", "rhr",
                         "hr_rest", "heart_rate_rest", "frequence_cardiaque_repos"],
    "sommeil_h": ["sommeil_h", "sommeil", "heures_sommeil", "sleep", "sleep_h", "sleep_hours", "hours_of_sleep",
                  "duree_sommeil", "sleep_duration"],
    "age": ["age", "ages", "age_years", "age_ans"],
    "imc": ["imc", "bmi", "body_mass_index", "indice_masse_corporelle"],
    "historique_blessures": ["historique_blessures", "antecedents", "antecedents_blessures", "previous_injuries",
                             "prior_injuries", "injury_history", "nb_blessures", "past_injuries", "blessures_passees"],
    "deficit_calorique_pct": ["deficit_calorique_pct", "deficit_calorique", "calorie_deficit", "caloric_deficit",
                              "energy_deficit", "ecart_calorique", "calorie_balance_pct", "calorie_balance",
                              "energy_balance", "bilan_calorique", "bilan_energetique"],
    "lactate": ["lactate", "lactate_mmol", "blood_lactate", "taux_lactate"],
    "intensite_moyenne_7j": ["intensite_moyenne_7j", "intensite", "intensite_7j", "intensity", "avg_intensity",
                             "mean_intensity_7d", "intensity_7d", "training_intensity"],
}
# Pour calculer l'IMC quand le fichier donne le poids et la taille mais pas l'IMC.
SYN_POIDS = ["poids", "poids_kg", "weight", "weight_kg", "masse", "masse_kg", "mass", "body_mass"]
SYN_TAILLE = ["taille", "taille_cm", "taille_m", "height", "height_cm", "height_m", "stature"]

# Variables qui portent l'essentiel de l'information du modèle (importance par permutation,
# rapport final, figure 6b). Sans aucune d'elles, les scores ne départagent presque pas les athlètes.
VARIABLES_CLES = ["sommeil_h", "historique_blessures", "acwr"]
SEUIL_HORS_DOMAINE = 0.20   # au-delà de 20 % de valeurs bornées, la variable sort du domaine appris
SEUIL_COLONNE_VIDE = 0.50   # au-delà de 50 % de valeurs illisibles/vides, la colonne ne compte pas comme mesurée
# Un BILAN (mangé - besoin) est négatif quand l'athlète mange trop peu : c'est l'opposé du déficit appris
# par le modèle (positif = mange trop peu). Ces colonnes sont donc lues avec le signe inversé.
SYN_BILAN_CALORIQUE = {"ecart_calorique", "calorie_balance_pct", "calorie_balance", "energy_balance",
                       "bilan_calorique", "bilan_energetique"}

SYN_ETIQUETTE = ["blessure", "blesse", "injury", "injured", "injury_occurred", "label", "target", "y", "cible", "is_injured"]
SYN_FATIGUE = ["fatigue", "fatigue_score", "fatigue_level", "niveau_fatigue", "score_fatigue"]
VRAI = {"1", "true", "vrai", "oui", "yes", "y", "o", "blesse", "injured", "injury", "positive", "pos"}
FAUX = {"0", "false", "faux", "non", "no", "n", "sain", "healthy", "none", "negative", "neg", "aucune"}


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


# ------------------------------------------------------------------------------------- lecture ----

def lire_csv(chemin: Path) -> pd.DataFrame:
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    derniere = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(chemin, sep=None, engine="python", encoding=enc)
            if df.shape[1] >= 2:
                return df
            df = pd.read_csv(chemin, sep=";", encoding=enc)
            return df
        except Exception as e:                       # noqa: BLE001 - on tente l'encodage suivant
            derniere = e
    raise ValueError(f"Lecture du CSV impossible : {derniere}")


def _en_numerique(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)
    txt = serie.astype(str).str.strip().str.replace(" ", "", regex=False).str.replace(" ", "", regex=False)
    txt = txt.str.replace(",", ".", regex=False).replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan,
                                                          "N/A": np.nan, "null": np.nan, "-": np.nan})
    return pd.to_numeric(txt, errors="coerce")


# ------------------------------------------------------------------------------ correspondance ----

def associer_colonnes(df: pd.DataFrame, mapping: dict | None = None) -> dict:
    """Retourne {variable_modèle: {'colonne': nom|None, 'methode': ...}}. Une colonne n'est utilisée qu'une fois."""
    colonnes = {c: _norm(c) for c in df.columns}
    inverse = {}
    for c, n in colonnes.items():
        inverse.setdefault(n, c)
    utilisees, resultat = set(), {}
    mapping = mapping or {}

    for feat in FEATURE_NAMES:
        if feat in mapping:
            src = mapping[feat]
            if src in df.columns:
                resultat[feat] = {"colonne": src, "methode": "mapping"}
                utilisees.add(src)
                continue
            raise ValueError(f"Le mapping associe {feat} à « {src} », colonne absente du fichier. "
                             f"Colonnes disponibles : {list(df.columns)}")
        resultat[feat] = {"colonne": None, "methode": "absente"}

    for feat in FEATURE_NAMES:                       # 1) noms exacts / synonymes
        if resultat[feat]["colonne"]:
            continue
        for syn in SYNONYMES[feat]:
            c = inverse.get(syn)
            if c is not None and c not in utilisees:
                resultat[feat] = {"colonne": c, "methode": "nom exact/synonyme"}
                utilisees.add(c)
                break
    restantes = {n: c for n, c in inverse.items() if c not in utilisees}
    for feat in FEATURE_NAMES:                       # 2) approximation prudente (similarité ≥ 0,85)
        if resultat[feat]["colonne"]:
            continue
        candidats = difflib.get_close_matches(feat, list(restantes), n=1, cutoff=0.85)
        for syn in SYNONYMES[feat]:
            candidats += difflib.get_close_matches(syn, list(restantes), n=1, cutoff=0.85)
        if candidats:
            c = restantes.pop(candidats[0])
            resultat[feat] = {"colonne": c, "methode": f"approximation ({candidats[0]})"}
            utilisees.add(c)
    resultat["_utilisees"] = utilisees
    return resultat


def deriver_imc(df: pd.DataFrame, assoc: dict) -> tuple[pd.DataFrame, list]:
    """Si l'IMC manque mais que le poids (kg) et la taille (cm ou m) sont présents, l'IMC est
    calculé (poids / taille²) et signalé comme tel. Une taille médiane > 3 est lue en cm."""
    notes = []
    if assoc["imc"]["colonne"]:
        return df, notes
    utilisees = set(assoc.get("_utilisees", set()))
    col_p = next((c for c in df.columns if c not in utilisees and _norm(c) in SYN_POIDS), None)
    col_t = next((c for c in df.columns if c not in utilisees and _norm(c) in SYN_TAILLE), None)
    if not (col_p and col_t):
        return df, notes
    poids, taille = _en_numerique(df[col_p]), _en_numerique(df[col_t])
    if taille.median() > 3:
        taille = taille / 100
    nom = f"IMC calculé ({col_p} / {col_t}²)"
    df = df.copy()
    df[nom] = (poids / taille ** 2).round(2)
    assoc["imc"] = {"colonne": nom, "methode": "calculé (poids / taille²)"}
    assoc.setdefault("_utilisees", set()).update({col_p, col_t, nom})
    notes.append(f"IMC calculé à partir de « {col_p} » et « {col_t} » (taille lue en "
                 f"{'cm' if _en_numerique(df[col_t]).median() > 3 else 'm'}).")
    return df, notes


def verdict_compatibilite(assoc: dict, qualite: dict, n_lignes: int, score: np.ndarray) -> dict:
    """Dit honnêtement si le fichier permet au modèle de produire des scores utiles.
    - incompatible : aucune des variables clés n'est mesurée, les scores sont quasi constants ;
    - partiel : moins de 5 variables, une variable clé manque, ou les données sortent du domaine appris ;
    - compatible : sinon."""
    # Une colonne trouvée mais vide ou illisible (ex. « 7h30 ») n'apporte rien : elle ne compte pas.
    vides = [f for f in FEATURE_NAMES
             if assoc[f]["colonne"] and qualite[f]["manquantes"] > SEUIL_COLONNE_VIDE * max(1, n_lignes)]
    mesurees = [f for f in FEATURE_NAMES if assoc[f]["colonne"] and f not in vides]
    cles_presentes = [f for f in VARIABLES_CLES if f in mesurees]
    hors_domaine = {f: round((qualite[f]["sous_domaine"] + qualite[f]["sur_domaine"]) / max(1, n_lignes), 3)
                    for f in mesurees}
    hors_domaine = {f: t for f, t in hors_domaine.items() if t > SEUIL_HORS_DOMAINE}
    dispersion = float(np.std(score)) if len(score) else 0.0
    raisons = []
    if not cles_presentes:
        statut = "incompatible"
        raisons.append("Aucune des variables qui portent l'essentiel de l'information du modèle "
                       f"({', '.join(LABELS_LISIBLES[f] for f in VARIABLES_CLES)}) n'est présente.")
    elif len(mesurees) < 5 or len(cles_presentes) < len(VARIABLES_CLES) or hors_domaine:
        statut = "partiel"
    else:
        statut = "compatible"
    for f in vides:
        raisons.append(f"{LABELS_LISIBLES[f]} : colonne « {assoc[f]['colonne']} » trouvée mais vide ou illisible "
                       f"à {qualite[f]['manquantes'] / max(1, n_lignes) * 100:.0f} % : comptée comme absente.")
    if len(mesurees) < 5:
        raisons.append(f"Seulement {len(mesurees)} variable(s) sur 10 réellement mesurée(s).")
    manquantes_cles = [LABELS_LISIBLES[f] for f in VARIABLES_CLES if f not in mesurees]
    if cles_presentes and manquantes_cles:
        raisons.append("Variable(s) clé(s) absente(s) : " + ", ".join(manquantes_cles) + ".")
    for f, t in hors_domaine.items():
        raisons.append(f"{LABELS_LISIBLES[f]} : {t * 100:.0f} % des valeurs hors du domaine appris "
                       f"{DOMAINE_ENTRAINEMENT[f]} (bornées) ; le modèle extrapole.")
    if dispersion < 0.02:
        raisons.append(f"Les scores varient à peine (écart-type {dispersion:.3f}) : le modèle ne départage "
                       "pas les athlètes de ce fichier.")
        if statut == "compatible":
            statut = "partiel"
    return {"statut": statut, "raisons": raisons, "variables_cles_presentes": cles_presentes,
            "hors_domaine": hors_domaine, "dispersion_scores": round(dispersion, 4), "colonnes_vides": vides}


def _trouver(df, nom_force, synonymes, exclues):
    if nom_force:
        if nom_force not in df.columns:
            raise ValueError(f"Colonne « {nom_force} » absente. Colonnes disponibles : {list(df.columns)}")
        return nom_force
    for c in df.columns:
        if c not in exclues and _norm(c) in synonymes:
            return c
    return None


def lire_etiquette(serie: pd.Series) -> tuple[np.ndarray, list]:
    """0/1 depuis 0/1, booléens, oui/non, injured/healthy… Refuse ce qu'il ne comprend pas."""
    notes = []
    s = serie.dropna()
    txt = s.astype(str).str.strip().str.lower()
    if set(txt.unique()) <= (VRAI | FAUX):
        y = serie.astype(str).str.strip().str.lower().map(lambda v: 1.0 if v in VRAI else (0.0 if v in FAUX else np.nan))
        return y.to_numpy(), notes
    num = _en_numerique(serie)
    valeurs = np.unique(num.dropna())
    if len(valeurs) == 2:
        notes.append(f"Étiquette à deux valeurs {valeurs.tolist()} : la plus grande est lue comme « blessure ».")
        return (num == valeurs.max()).where(num.notna()).astype(float).to_numpy(), notes
    if len(valeurs) > 2 and valeurs.min() >= 0 and num.notna().sum() >= 0.5 * len(s):
        # Ex. nombre de blessures (0, 1, 2, 3) ou jours d'arrêt : plutôt que de tout refuser,
        # « au moins une blessure » = 1. C'est dit dans le rapport.
        notes.append(f"Étiquette à {len(valeurs)} valeurs ({', '.join(f'{v:g}' for v in valeurs[:6])}"
                     f"{'…' if len(valeurs) > 6 else ''}) : toute valeur > 0 est lue comme « blessure ».")
        return (num > 0).where(num.notna()).astype(float).to_numpy(), notes
    raise ValueError("Colonne d'étiquette non reconnue : attendu 0/1, oui/non, true/false, injured/healthy. "
                     f"Valeurs vues : {sorted(map(str, txt.unique()))[:8]}")


# --------------------------------------------------------------------------------------- nettoyage ----

def nettoyer_matrice(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Même règle que models.injury_risk.valider_et_nettoyer_features, vectorisée.
    NaN -> valeur typique de l'entraînement ; hors domaine -> borne. Retourne aussi les compteurs."""
    Xn = X.astype(float).copy()
    stats = {}
    for j, nom in enumerate(FEATURE_NAMES):
        lo, hi = DOMAINE_ENTRAINEMENT[nom]
        col = Xn[:, j]
        manq = ~np.isfinite(col)
        bas = np.isfinite(col) & (col < lo)
        haut = np.isfinite(col) & (col > hi)
        col[manq] = VALEURS_TYPIQUES[nom]
        col[bas] = lo
        col[haut] = hi
        stats[nom] = {"manquantes": int(manq.sum()), "sous_domaine": int(bas.sum()), "sur_domaine": int(haut.sum())}
    return Xn, stats


def construire_X(df: pd.DataFrame, assoc: dict) -> np.ndarray:
    X = np.full((len(df), len(FEATURE_NAMES)), np.nan)
    for j, feat in enumerate(FEATURE_NAMES):
        c = assoc[feat]["colonne"]
        if c is not None:
            X[:, j] = _en_numerique(df[c]).to_numpy()
    return X


def convertir_unites(X: np.ndarray, assoc: dict) -> tuple[np.ndarray, list]:
    """Corrige les unités les plus courantes d'un fichier inconnu, en le disant à chaque fois.
    Les seuils sont choisis pour ne jamais se déclencher sur des valeurs déjà dans la bonne unité."""
    X = X.copy()
    notes = []

    def _col(feat):
        return FEATURE_NAMES.index(feat)

    def _mediane(j):
        v = X[:, j][np.isfinite(X[:, j])]
        return float(np.median(v)) if len(v) else None, (float(np.max(np.abs(v))) if len(v) else None)

    j = _col("sommeil_h")
    med, _ = _mediane(j)
    if assoc["sommeil_h"]["colonne"] and med is not None and med > 24:
        X[:, j] /= 60
        notes.append(f"Sommeil lu en minutes (médiane {med:g}) : converti en heures.")

    j = _col("intensite_moyenne_7j")
    med, maxi = _mediane(j)
    if assoc["intensite_moyenne_7j"]["colonne"] and med is not None and med > 1.5:
        div = 10 if maxi <= 10.5 else 100
        X[:, j] /= div
        notes.append(f"Intensité lue sur une échelle 0-{div} (médiane {med:g}) : ramenée à 0-1.")

    j = _col("deficit_calorique_pct")
    col = assoc["deficit_calorique_pct"]["colonne"]
    if col:
        if _norm(col) in SYN_BILAN_CALORIQUE:
            X[:, j] *= -1
            notes.append(f"« {col} » est un bilan (mangé - besoin) : signe inversé pour obtenir un déficit "
                         "(positif = mange moins que le besoin).")
        med, maxi = _mediane(j)
        if maxi is not None and 0 < maxi <= 1:
            X[:, j] *= 100
            notes.append("Écart calorique donné en fraction (ex. 0,15) : converti en pourcentage.")
    return X, notes


# ---------------------------------------------------------------------------------------- métriques ----

def auc_bootstrap(y, s, n_boot=1000, graine=0):
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return None
    auc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(graine)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return {"auc": round(auc, 4), "ic95": [round(float(lo), 4), round(float(hi), 4)]}


def metriques_classification(y, score, proba_corr) -> dict:
    y, score = np.asarray(y).astype(int), np.asarray(score)
    out = {"n": int(len(y)), "prevalence": round(float(y.mean()), 4)}
    a = auc_bootstrap(y, score)
    if a is None:
        out["avertissement"] = "Une seule classe dans l'étiquette : AUC non définie."
        return out
    out.update(a)
    pred = score >= 0.5
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out["seuil_0_5"] = {"precision": round(tp / (tp + fp), 4) if tp + fp else None,
                        "rappel": round(tp / (tp + fn), 4) if tp + fn else None,
                        "signales": int(pred.sum())}
    k = max(1, int(round(0.2 * len(y))))
    haut = np.argsort(-score)[:k]
    rappel20 = float(y[haut].sum() / max(1, y.sum()))
    out["top20pct"] = {"rappel": round(rappel20, 4), "lift": round(rappel20 / 0.2, 2),
                       "precision": round(float(y[haut].mean()), 4)}
    out["brier_proba_corrigee"] = round(float(brier_score_loss(y, proba_corr)), 4)
    out["brier_constante"] = round(float(brier_score_loss(y, np.full(len(y), y.mean()))), 4)
    if len(y) >= 50:
        q = np.quantile(proba_corr, [0, .2, .4, .6, .8, 1.0])
        tranches = []
        for i in range(5):
            m = (proba_corr >= q[i]) & ((proba_corr <= q[i + 1]) if i == 4 else (proba_corr < q[i + 1]))
            if m.sum():
                tranches.append({"tranche": i + 1, "n": int(m.sum()), "proba_moyenne": round(float(proba_corr[m].mean()), 3),
                                 "frequence_observee": round(float(y[m].mean()), 3)})
        out["fiabilite"] = tranches
    if len(y) < 200 or y.sum() < 20 or (len(y) - y.sum()) < 20:
        out["avertissement"] = "Échantillon petit (moins de 200 lignes ou moins de 20 cas par classe) : l'IC95 est large, ne pas sur-interpréter."
    return out


def metriques_regression(y, p) -> dict:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return {"n": int(len(y)), "avertissement": "Trop peu de valeurs de fatigue."}
    return {"n": int(len(y)), "mae": round(float(mean_absolute_error(y, p)), 4),
            "rmse": round(float(mean_squared_error(y, p) ** 0.5), 4), "r2": round(float(r2_score(y, p)), 4)}


# ----------------------------------------------------------------------------- réentraînement (option) ----

def modeles_etude(Xtr: np.ndarray, ytr: np.ndarray, colonnes: list) -> dict:
    """Modèles de l'étude, réentraînés sur le CSV d'entraînement fourni (variables disponibles seulement)."""
    poids = compute_sample_weight("balanced", ytr)
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(Xtr[:, colonnes])
    mods = {"GB d = 1 (optimal si risque additif, E2)": GradientBoostingClassifier(n_estimators=180, max_depth=1, learning_rate=0.08, random_state=0),
            "GB d = 2 (choix robuste, E3)": GradientBoostingClassifier(n_estimators=180, max_depth=2, learning_rate=0.08, random_state=0),
            "GB d = 3 (référence E1)": GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.08, random_state=0)}
    for m in mods.values():
        m.fit(Xi, ytr, sample_weight=poids)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    lr.fit(Xi, ytr, logisticregression__sample_weight=poids)
    mods["Régression logistique"] = lr
    return {"imputeur": imp, "modeles": mods, "colonnes": colonnes}


def scores_etude(bundle: dict, Xte: np.ndarray) -> dict:
    Xi = bundle["imputeur"].transform(Xte[:, bundle["colonnes"]])
    return {nom: m.predict_proba(Xi)[:, 1] for nom, m in bundle["modeles"].items()}


# ---------------------------------------------------------------------------------------- pipeline ----

def analyser(chemin_test, cible=None, fatigue=None, mapping=None, chemin_train=None, n_boot=1000) -> dict:
    df = lire_csv(chemin_test)
    assoc = associer_colonnes(df, mapping)
    df, notes_imc = deriver_imc(df, assoc)
    exclues = set(assoc.pop("_utilisees"))
    col_y = _trouver(df, cible, SYN_ETIQUETTE, exclues)
    exclues |= {col_y} if col_y else set()
    col_f = _trouver(df, fatigue, SYN_FATIGUE, exclues)

    X, notes_unites = convertir_unites(construire_X(df, assoc), assoc)
    n_mesurees = sum(1 for f in FEATURE_NAMES if assoc[f]["colonne"])
    if n_mesurees == 0:
        raise ValueError("Aucune des 10 variables du modèle n'a été trouvée dans le fichier. "
                         f"Colonnes vues : {list(df.columns)}. Fournir --mapping (ex. {{\"sommeil_h\": \"sleep\"}}).")
    Xn, stats = nettoyer_matrice(X)
    clf, reg, _ = _charger_modeles()
    score = clf.predict_proba(Xn)[:, 1]
    proba = np.array([corriger_prevalence(float(s)) for s in score])
    fat = np.clip(reg.predict(Xn), 0, 10)

    res = {"fichier": str(chemin_test), "date": datetime.now().isoformat(timespec="seconds"),
           "lignes": int(len(df)), "colonnes_fichier": list(map(str, df.columns)),
           "correspondance": {f: assoc[f] for f in FEATURE_NAMES}, "variables_mesurees": n_mesurees,
           "qualite": stats, "avertissements": list(notes_imc) + notes_unites}
    res["verdict"] = verdict_compatibilite(assoc, stats, len(df), score)
    absentes = [f for f in FEATURE_NAMES if not assoc[f]["colonne"]]
    if absentes:
        res["avertissements"].append("Variables absentes, remplacées par leur valeur typique à l'entraînement : "
                                     + ", ".join(f"{LABELS_LISIBLES[f]} ({VALEURS_TYPIQUES[f]:g})" for f in absentes)
                                     + ".")
    if n_mesurees < 5:
        res["avertissements"].append("Moins de 5 variables sur 10 présentes : les scores sont peu informatifs.")
    taux_manq = float(np.isnan(X[:, [j for j, f in enumerate(FEATURE_NAMES) if assoc[f]['colonne']]]).mean())
    res["taux_valeurs_manquantes_colonnes_presentes"] = round(taux_manq, 4)
    doublons = int(df.duplicated().sum())
    if doublons:
        res["avertissements"].append(f"{doublons} lignes dupliquées dans le fichier (conservées).")

    sortie_df = df.copy()
    sortie_df["athena_score_risque"] = np.round(score, 4)
    sortie_df["athena_proba_corrigee"] = np.round(proba, 4)
    sortie_df["athena_fatigue"] = np.round(fat, 2)
    res["_predictions"] = sortie_df

    if col_y:
        y, notes = lire_etiquette(df[col_y])
        res["avertissements"] += notes
        ok = np.isfinite(y)
        if not ok.all():
            res["avertissements"].append(f"{int((~ok).sum())} lignes sans étiquette ignorées pour les métriques.")
        res["colonne_etiquette"] = col_y
        res["production"] = metriques_classification(y[ok], score[ok], proba[ok])
        yy = y[ok]
        res["_y"] = yy
    if col_f:
        res["colonne_fatigue"] = col_f
        res["fatigue"] = metriques_regression(_en_numerique(df[col_f]).to_numpy(), fat)

    if chemin_train:
        dtr = lire_csv(chemin_train)
        atr = associer_colonnes(dtr, mapping)
        atr.pop("_utilisees")
        cy_tr = _trouver(dtr, cible, SYN_ETIQUETTE, {atr[f]["colonne"] for f in FEATURE_NAMES if atr[f]["colonne"]})
        if not cy_tr:
            raise ValueError("Le fichier d'entraînement doit contenir une colonne d'étiquette (--cible).")
        ytr, _ = lire_etiquette(dtr[cy_tr])
        okt = np.isfinite(ytr)
        Xtr = convertir_unites(construire_X(dtr, atr), atr)[0][okt]
        communes = [j for j, f in enumerate(FEATURE_NAMES) if atr[f]["colonne"] and assoc[f]["colonne"]]
        if not communes:
            raise ValueError("Aucune variable commune entre entraînement et test.")
        bundle = modeles_etude(Xtr, ytr[okt].astype(int), communes)
        sc = scores_etude(bundle, X)
        comp = {}
        if col_y:
            comp["Production (modèle déjà entraîné)"] = auc_bootstrap(res["_y"], score[ok], n_boot)
            for nom, s in sc.items():
                comp[nom] = auc_bootstrap(res["_y"], s[ok], n_boot)
        res["reentrainement"] = {"n_train": int(okt.sum()), "variables": [FEATURE_NAMES[j] for j in communes], "auc": comp}
        for nom, s in sc.items():
            sortie_df["athena_score_" + _norm(nom)[:18]] = np.round(s, 4)
    return res


# -------------------------------------------------------------------------------------------- rapport ----

def _f(x, nd=3):
    return "n/d" if x is None else f"{x:.{nd}f}".replace(".", ",")


def rapport_md(res: dict) -> str:
    L = [f"# Rapport de test ATHÉNA — {Path(res['fichier']).name}", "",
         f"Date : {res['date']} · lignes : {res['lignes']} · variables du modèle trouvées : {res['variables_mesurees']}/10", ""]
    v = res.get("verdict")
    if v:
        L += [f"## Verdict : fichier {v['statut'].upper()}", ""]
        L += [f"- {r}" for r in v["raisons"]] or ["- Toutes les variables clés sont présentes et dans le domaine appris."]
        L.append("")
    L += ["## Correspondance des colonnes", "", "| Variable du modèle | Colonne du fichier | Méthode |", "|---|---|---|"]
    for f in FEATURE_NAMES:
        a = res["correspondance"][f]
        L.append(f"| {LABELS_LISIBLES[f]} | {a['colonne'] or '— (absente)'} | {a['methode']} |")
    L += ["", "## Qualité des données (corrections appliquées comme en production)", "",
          "| Variable | Manquantes | Sous le domaine | Au-dessus du domaine |", "|---|---|---|---|"]
    for f in FEATURE_NAMES:
        s = res["qualite"][f]
        L.append(f"| {LABELS_LISIBLES[f]} | {s['manquantes']} | {s['sous_domaine']} | {s['sur_domaine']} |")
    for a in res["avertissements"]:
        L.append(f"\n⚠️ {a}")
    if "production" in res:
        p = res["production"]
        L += ["", f"## Risque de blessure — modèle de production (étiquette : « {res['colonne_etiquette']} »)", ""]
        if "auc" in p:
            L.append(f"- **ROC-AUC = {_f(p['auc'])}** (IC95 bootstrap [{_f(p['ic95'][0])} ; {_f(p['ic95'][1])}]) — n = {p['n']}, prévalence {_f(p['prevalence'] * 100, 1)} %")
            s = p["seuil_0_5"]
            L.append(f"- Seuil 0,5 : précision {_f(s['precision'])} · rappel {_f(s['rappel'])} · {s['signales']} profils signalés")
            t = p["top20pct"]
            L.append(f"- 20 % des profils les plus à risque : rappel {_f(t['rappel'])} (gain {t['lift']}× par rapport au hasard) · précision {_f(t['precision'])}")
            L.append(f"- Brier de la probabilité corrigée {_f(p['brier_proba_corrigee'], 4)} (référence constante {_f(p['brier_constante'], 4)})")
            if "fiabilite" in p:
                L += ["", "| Tranche de risque | n | Probabilité prédite | Fréquence observée |", "|---|---|---|---|"]
                for t in p["fiabilite"]:
                    L.append(f"| {t['tranche']} | {t['n']} | {_f(t['proba_moyenne'])} | {_f(t['frequence_observee'])} |")
        if "avertissement" in p:
            L.append(f"\n⚠️ {p['avertissement']}")
    if "fatigue" in res:
        r = res["fatigue"]
        L += ["", f"## Fatigue (colonne « {res['colonne_fatigue']} »)", ""]
        L.append(f"- MAE {_f(r.get('mae'))} · RMSE {_f(r.get('rmse'))} · R² {_f(r.get('r2'))} (n = {r['n']})" if "mae" in r else f"⚠️ {r['avertissement']}")
    if "reentrainement" in res:
        r = res["reentrainement"]
        L += ["", f"## Modèles réentraînés sur le fichier d'entraînement (n = {r['n_train']})", "",
              "| Modèle | ROC-AUC | IC95 |", "|---|---|---|"]
        for nom, a in r["auc"].items():
            L.append(f"| {nom} | {_f(a['auc'])} | [{_f(a['ic95'][0])} ; {_f(a['ic95'][1])}] |" if a else f"| {nom} | n/d | |")
        L.append(f"\nVariables utilisées pour le réentraînement : {', '.join(r['variables'])}.")
    L += ["", "---", "Outil d'aide à l'analyse : ces scores ne sont pas un diagnostic médical."]
    return "\n".join(L)


def ecrire_sorties(res: dict, dossier: Path) -> Path:
    dossier = Path(dossier) / datetime.now().strftime("%Y%m%d_%H%M%S")
    dossier.mkdir(parents=True, exist_ok=True)
    res["_predictions"].to_csv(dossier / "predictions.csv", index=False)
    resume = {k: v for k, v in res.items() if not k.startswith("_")}
    (dossier / "resume.json").write_text(json.dumps(resume, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (dossier / "rapport.md").write_text(rapport_md(res), encoding="utf-8")
    return dossier


# ------------------------------------------------------------------------------------ répétition générale ----

def jeu_de_repetition(dossier: Path, n=3000, graine=123) -> Path:
    """Jeu inédit simulé, colonnes en anglais dans le désordre, séparateur « ; », virgule décimale,
    8 % de valeurs manquantes et quelques valeurs aberrantes : simule un fichier de jury."""
    X, y, fat = _generer_dataset(n=n, graine=graine)
    noms = {"acwr": "ACWR", "charge_chronique": "Chronic Load", "fc_repos_estimee": "Resting HR", "sommeil_h": "Sleep (h)",
            "age": "Age", "imc": "BMI", "historique_blessures": "Previous injuries", "deficit_calorique_pct": "Calorie deficit %",
            "lactate": "Lactate", "intensite_moyenne_7j": "Avg intensity 7d"}
    df = pd.DataFrame(X, columns=FEATURE_NAMES).rename(columns=noms)
    rng = np.random.default_rng(graine)
    for c in df.columns:
        df.loc[rng.random(n) < 0.08, c] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "Sleep (h)"] = 25.0
    df["Injured"] = np.where(y == 1, "yes", "no")
    df["Fatigue score"] = fat
    df["player_id"] = [f"P{i:04d}" for i in range(n)]
    df = df[list(rng.permutation(df.columns))]
    chemin = Path(dossier) / "repetition_inedit.csv"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(chemin, sep=";", decimal=",", index=False)
    return chemin


def main(argv=None):
    # La console Windows (cp1252) ne sait pas afficher « ⚠️ » ou « ² » : sans cela, le rapport
    # plantait à l'affichage, après le calcul (le jour J, devant le jury).
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Test en direct d'ATHÉNA sur un CSV inédit.")
    ap.add_argument("--test", help="CSV inédit à évaluer")
    ap.add_argument("--train", help="CSV d'entraînement (optionnel) : réentraîne les modèles de l'étude")
    ap.add_argument("--cible", help="nom de la colonne d'étiquette (blessure) si non détectée")
    ap.add_argument("--fatigue", help="nom de la colonne de fatigue si non détectée")
    ap.add_argument("--mapping", help="JSON {variable_modèle: colonne_du_fichier}")
    ap.add_argument("--sortie", default="sorties_test", help="dossier de sortie")
    ap.add_argument("--demo", action="store_true", help="répétition générale sur un jeu inédit simulé")
    a = ap.parse_args(argv)

    if a.demo:
        a.test = str(jeu_de_repetition(Path(a.sortie)))
        print(f"[démo] fichier simulé écrit : {a.test}\n")
    if not a.test:
        ap.error("--test est requis (ou --demo).")
    mapping = json.loads(Path(a.mapping).read_text(encoding="utf-8")) if a.mapping else None
    try:
        res = analyser(a.test, a.cible, a.fatigue, mapping, a.train)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERREUR : {e}")
        return 2
    dossier = ecrire_sorties(res, Path(a.sortie))
    print(rapport_md(res))
    print(f"\nFichiers écrits dans {dossier}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
