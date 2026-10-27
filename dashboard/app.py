# ATHÉNA v2 — Plateforme IA sportive prédictive
# Développée par ARiA Tech pour la Pan-African Robotics Competition (PARC 2026)
# Ligue Intelligence Artificielle
# Lancement : streamlit run dashboard/app.py   (depuis la racine du projet)

import hashlib
import shutil
import sys
import tempfile
import zlib
from pathlib import Path
from collections import deque
from datetime import datetime

import streamlit as st

# Rend le package "athena" importable quel que soit le répertoire de lancement
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.theme import (  # noqa: E402 - léger : le thème s'affiche avant les imports lourds
    PALETTE, MODELE_PLOTLY, injecter_theme, marque_barre_laterale, en_tete,
    carte_kpi, carte_alerte, pied_de_page,
)
from dashboard.ecran_demarrage import EcranDemarrage  # noqa: E402

# Version publique sur Streamlit Community Cloud (l'application y est installée sous /mount/src).
EN_LIGNE = str(Path(__file__).resolve()).startswith("/mount/src")


# CONFIGURATION GÉNÉRALE
st.set_page_config(
    page_title="ATHÉNA | ARiA Tech - PARC 2026",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="auto",   # fermée sur téléphone (QR code du poster), ouverte sur ordinateur
)
injecter_theme()

# ÉCRAN DE DÉMARRAGE : seulement au tout premier chargement, quand les modules lourds ne sont pas
# encore en mémoire (~5 à 8 s). Ensuite tout est déjà chargé : aucune attente artificielle.
demarrage = EcranDemarrage() if "models.injury_risk" not in sys.modules else None


def _etape_faite():
    if demarrage:
        demarrage.valider()


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.express as px  # noqa: E402
_etape_faite()   # Moteur de calcul

from models.injury_risk import (  # noqa: E402
    construire_vecteur_features, corriger_prevalence, predire_profil_risque, predire_tendance_risque,
    _charger_modeles,
)
from models.experiment_comparaison import charger_ou_executer_experience  # noqa: E402
from models.explicabilite import expliquer_prediction, recommander_leviers, simuler_scenario  # noqa: E402
_charger_modeles()
_etape_faite()   # Modèle de risque de blessure

from data.simulators import (  # noqa: E402
    generer_signal_ecg, simuler_constantes_vitales, generer_historique_charge,
    generer_micronutriments_simules, generer_snapshot_equipe, EQUIPE_NATIONALE_DEMO,
)
from analytics.metrics import (  # noqa: E402
    calculer_acwr, calculer_bmr_tdee, moteur_de_decision, calculer_performance_score,
)
from analytics.ai_coach import generer_recommandations_coach  # noqa: E402
from reports.pdf_report import generer_rapport_pdf  # noqa: E402
from data.donnees_reelles import (  # noqa: E402
    COLONNES as COLONNES_JOURNAL, DESCRIPTION_COLONNES, charger_journal, ajouter_entree,
    importer_csv, gabarit_csv, risque_actuel, evaluer_sur_donnees_reelles, DOSSIER_REEL,
    ErreurJournal, ajouter_seance, charge_28j, moyennes_7j, charger_repas, ajouter_repas, supprimer_repas,
    COLONNES_REPAS,
)
_etape_faite()   # Coach IA, moteur de décision et rapports

try:
    # Préchauffage du détecteur de pose (~2 s) : la 1re analyse vidéo devant le jury est immédiate.
    from computer_vision.pose_analysis import modele_disponible_localement as _pose_locale, _charger_detecteur
    if _pose_locale() or EN_LIGNE:   # en ligne : le modèle (9 Mo, absent de git) est téléchargé ici
        _charger_detecteur()
except Exception:  # noqa: BLE001 - l'onglet Analyse Vidéo affichera l'erreur détaillée
    pass
_etape_faite()   # Détecteur de posture


def _pleine_largeur(fn, *args, **kwargs):
    """Affiche un élément en pleine largeur sur TOUTES les versions de Streamlit.

    `use_container_width` est déprécié depuis fin 2025 et retiré des versions
    récentes (remplacé par `width="stretch"`), alors que `width="stretch"`
    n'existe pas dans les versions plus anciennes : on essaie les deux.
    """
    derniere_erreur = None
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return fn(*args, **kwargs, **extra)
        except Exception as e:  # noqa: BLE001 - on retente avec l'autre API
            derniere_erreur = e
    raise derniere_erreur


def tracer(fig):
    return _pleine_largeur(st.plotly_chart, fig)


def tableau(df):
    return _pleine_largeur(st.dataframe, df)


def image(img, **kw):
    return _pleine_largeur(st.image, img, **kw)


def _flash(zone: str, message: str, genre: str = "success"):
    """Garde un message pour l'exécution suivante : st.rerun() efface sinon tout ce qui
    vient d'être affiché (d'où l'impression que « rien ne se passe » après un enregistrement)."""
    st.session_state[f"flash_{zone}"] = (genre, message)


def _afficher_flash(zone: str):
    if f"flash_{zone}" in st.session_state:
        genre, message = st.session_state.pop(f"flash_{zone}")
        getattr(st, genre)(message)
        st.toast(message)


@st.cache_data(show_spinner="Chargement de l'étude scientifique…")
def etude_scientifique_en_cache() -> dict:
    """Résultats pré-calculés de l'étude (lus une seule fois par session)."""
    return charger_ou_executer_experience()


RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_FIGURES_RAPPORT = RACINE_PROJET / "reports" / "rapport_figures"


@st.cache_data
def experiences_e2_e3_en_cache() -> tuple:
    """Résultats de E2 (profondeur) et E3 (interaction), tels que publiés dans le rapport.
    Lecture seule des JSON pré-calculés : rien n'est relancé depuis le tableau de bord."""
    import json
    resultats = []
    for nom in ["experiment_profondeur.json", "experiment_interaction.json"]:
        chemin = RACINE_PROJET / "models" / "artifacts" / nom
        resultats.append(json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else None)
    return tuple(resultats)



# ÉTAT DE SESSION
for cle, defaut in [
    ("historique_vitals", deque(maxlen=200)),
    ("dernier_etat_vital", None),
    ("simulation_active", False),
    ("seances_simulees", []),   # séances ajoutées en mode simulation (non écrites sur disque)
    ("micronutriments", None),
]:
    if cle not in st.session_state:
        st.session_state[cle] = defaut

if st.session_state.micronutriments is None:
    st.session_state.micronutriments = generer_micronutriments_simules()


# BARRE LATÉRALE
with st.sidebar:
    marque_barre_laterale()

    st.subheader("Profil Athlète")
    # Choisir un athlète de l'effectif charge SON profil (le même que dans l'onglet
    # National Team). Les clés des widgets dépendent de l'athlète : changer d'athlète
    # recharge ses valeurs, et le staff peut toujours les ajuster à la main ensuite.
    profils_effectif = {a["nom"]: a for a in EQUIPE_NATIONALE_DEMO}
    liste_athletes = [f"{a['nom']} ({a['sport']})" for a in EQUIPE_NATIONALE_DEMO] + ["Profil personnalisé"]
    athlete_selectionne = st.selectbox("Sélectionner un athlète", liste_athletes)
    nom_athlete = athlete_selectionne.split(" (")[0]
    ref = profils_effectif.get(nom_athlete, {
        "age": 22, "poids": 68.0, "taille": 175.0, "sommeil_h": 7.0,
        "historique_blessures": 0, "intensite_effort": 0.5,
    })
    cle = nom_athlete.replace(" ", "_")

    # Source de la charge, des repas et du sommeil : la simulation de démo, ou le journal
    # réel d'un athlète (onglet Données Réelles). Les séances saisies y sont alors écrites.
    try:
        journal_global, erreur_journal = charger_journal(), None
    except ErreurJournal as err:
        journal_global, erreur_journal = pd.DataFrame(columns=COLONNES_JOURNAL), str(err)
    ids_journal = sorted(journal_global["athlete_id"].unique()) if len(journal_global) else []
    if st.session_state.get("source_donnees") not in [None, "Simulation (démo)", *ids_journal]:
        del st.session_state["source_donnees"]
    source_donnees = st.selectbox(
        "Données d'entraînement", ["Simulation (démo)"] + ids_journal, key="source_donnees",
        format_func=lambda s: s if s.startswith("Simulation") else f"Journal réel : {s}",
        help="Choisis un identifiant du journal (onglet Données Réelles) pour utiliser ses vraies "
             "séances : l'ACWR, le sommeil et la FC de repos en sont alors calculés.",
    )
    id_reel = None if source_donnees.startswith("Simulation") else source_donnees
    if erreur_journal:
        st.error(erreur_journal)
    moy_reelles = moyennes_7j(journal_global, id_reel) if id_reel else {}
    cle_mesures = cle if id_reel is None else f"{cle}_{id_reel}"

    sexe = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
    age = st.number_input("Âge", min_value=10, max_value=60, value=int(ref["age"]), key=f"age_{cle}")
    poids = st.number_input("Poids (kg)", min_value=30.0, max_value=150.0, value=float(ref["poids"]), key=f"poids_{cle}")
    taille = st.number_input("Taille (cm)", min_value=120.0, max_value=220.0, value=float(ref["taille"]), key=f"taille_{cle}")

    st.subheader("Données pour l'IA prédictive")
    # Avec un journal réel : moyenne des 7 derniers jours saisis (si disponible), bornée aux plages des widgets.
    fc_defaut = moy_reelles.get("fc_repos", np.nan)
    fc_defaut = int(np.clip(round(fc_defaut), 35, 100)) if np.isfinite(fc_defaut) else 60
    sommeil_defaut = moy_reelles.get("sommeil_h", np.nan)
    sommeil_defaut = float(np.clip(round(sommeil_defaut, 1), 3.0, 10.0)) if np.isfinite(sommeil_defaut) else float(ref["sommeil_h"])
    fc_repos = st.number_input("FC de repos (bpm)", min_value=35, max_value=100, value=fc_defaut, key=f"fc_{cle_mesures}")
    sommeil_h = st.slider("Sommeil moyen (h/nuit)", 3.0, 10.0, sommeil_defaut, 0.1, key=f"sommeil_{cle_mesures}")
    historique_blessures = st.number_input("Blessures (12 derniers mois)", min_value=0, max_value=10,
                                           value=int(ref["historique_blessures"]), key=f"blessures_{cle}")

    st.subheader("Mode d'acquisition")
    mode_acquisition = st.radio(
        "Source des données", ["Simulation", "Capteurs Bluetooth (live)"],
        help="Le mode Live nécessite un module physique connecté (point d'injection matériel prévu pour Raspberry Pi).",
    )
    if mode_acquisition != "Simulation":
        st.warning("Aucun capteur physique détecté. Basculement automatique sur le simulateur.")
    mode_simulation = mode_acquisition == "Simulation"

    st.subheader("Paramètres")
    intensite_effort = st.slider("Intensité d'effort (moyenne 7 j)", 0.0, 1.0, float(ref["intensite_effort"]), 0.05,
                                 key=f"intensite_{cle}")
    intervalle_rafraichissement = st.slider("Fréquence de rafraîchissement (s)", 0.2, 2.0, 0.6, 0.1)
    facteur_activite = st.select_slider(
        "Niveau d'activité (TDEE)", options=[1.2, 1.375, 1.55, 1.725, 1.9], value=1.55,
        format_func=lambda x: {1.2: "Sédentaire", 1.375: "Léger", 1.55: "Modéré",
                                1.725: "Intense", 1.9: "Athlète élite"}[x],
    )

profil_athlete = {
    "nom": athlete_selectionne, "sexe": sexe, "age": age, "poids": poids, "taille": taille,
    "intensite_effort": intensite_effort,
}


# CALCULS PARTAGÉS (utilisés par plusieurs onglets)
aujourdhui = datetime.now().date()


def charge_simulee(nom: str) -> pd.DataFrame:
    """Historique simulé propre à chaque athlète et STABLE (même graine à chaque rafraîchissement),
    plus les séances ajoutées pendant la session."""
    df = generer_historique_charge(graine=zlib.crc32(nom.encode("utf-8")))
    ajouts = [s for s in st.session_state.seances_simulees if s["athlete"] == nom]
    if ajouts:
        df = pd.concat([df, pd.DataFrame(ajouts)[["date", "charge"]]], ignore_index=True)
    return df


df_charge = charge_28j(journal_global, id_reel) if id_reel else charge_simulee(nom_athlete)
acwr_info = calculer_acwr(df_charge)
bmr_tdee = calculer_bmr_tdee(sexe, poids, taille, age, facteur_activite)
# Repas du jour, enregistrés sur disque pour l'athlète courant (identifiant du journal, sinon nom).
athlete_courant = id_reel or nom_athlete
try:
    repas_du_jour, erreur_repas = charger_repas(athlete_courant, aujourdhui), None
except ErreurJournal as err:
    repas_du_jour, erreur_repas = pd.DataFrame(columns=COLONNES_REPAS), str(err)
kcal_consommees = float(repas_du_jour["kcal"].sum()) if len(repas_du_jour) else 0
bilan_calorique = {"tdee": bmr_tdee["tdee"], "kcal_consommees": kcal_consommees,
                   "renseigne": len(repas_du_jour) > 0}
df_micro = st.session_state.micronutriments

vitals_courants = st.session_state.dernier_etat_vital or simuler_constantes_vitales(profil_athlete)
imc = poids / ((taille / 100) ** 2)
# Convention du modèle (train_injury_model) : déficit POSITIF quand l'athlète mange MOINS que son besoin.
# L'ancien calcul (mangé - besoin) inversait le signe : manger moins faisait baisser le risque affiché.
deficit_calorique_pct = (
    (bilan_calorique["tdee"] - bilan_calorique["kcal_consommees"]) / bilan_calorique["tdee"] * 100
    if bilan_calorique["tdee"] > 0 and bilan_calorique["renseigne"] else 0
)

features = construire_vecteur_features(
    acwr=acwr_info["acwr"], charge_chronique=acwr_info["chronique"], fc_repos=fc_repos,
    sommeil_h=sommeil_h, age=age, imc=imc, historique_blessures=historique_blessures,
    deficit_calorique_pct=deficit_calorique_pct, lactate=vitals_courants["lactate"],
    intensite_moyenne_7j=intensite_effort,
)
profil_risque = predire_profil_risque(features)
tendance_trajectoire, tendance_sens = predire_tendance_risque(features, df_charge)
recommandations = generer_recommandations_coach(profil_risque, acwr_info, bilan_calorique, sommeil_h)
explication_individuelle = expliquer_prediction(features)
plan_action = recommander_leviers(features)

df_charge_semaine = df_charge.tail(28)
regularite_pct = (
    min(100, (df_charge_semaine["charge"] > 150).sum() / len(df_charge_semaine) * 100)
    if len(df_charge_semaine) > 0 else 0
)
performance = calculer_performance_score(
    acwr_info, profil_risque["injury_risk_pct"], profil_risque["fatigue_score"],
    bilan_calorique, regularite_pct,
)

alertes = moteur_de_decision(vitals_courants, acwr_info, bilan_calorique, df_micro)
_etape_faite()   # Profil de l'athlète, risque et alertes


# EN-TÊTE (après les calculs : il affiche le statut de l'athlète)
en_tete(nom_athlete, profil_risque, acwr_info, mode_simulation,
        source=f"Données réelles : {id_reel}" if id_reel else "Simulation")

# Version publique : le disque est partagé par tous les visiteurs et effacé à chaque redémarrage.
if EN_LIGNE:
    st.info("Version de démonstration en ligne : les données saisies sont visibles par tous les visiteurs et "
            "effacées au redémarrage. Ne saisissez aucune donnée réelle d'athlète.")


# ONGLETS PRINCIPAUX
onglets = st.tabs([
    "Démo Guidée", "IA Prédictive", "Monitoring", "Charge & ACWR", "Nutrition",
    "Décisions", "Analyse Vidéo", "National Team", "Étude Scientifique", "Données Réelles", "Test CSV",
])
(onglet10, onglet5, onglet1, onglet2, onglet3, onglet4, onglet6, onglet7, onglet8, onglet9, onglet11) = onglets

# ONGLET 1 : MONITORING TEMPS RÉEL
with onglet1:
    st.subheader("Monitoring physiologique en temps réel")
    demarrer = st.toggle("Démarrer la simulation live", value=st.session_state.simulation_active)
    st.session_state.simulation_active = demarrer

    def afficher_etat_vital(vitals: dict):
        # Conteneur "vitaux" : style compact propre au Monitoring (voir .st-key-vitaux dans theme.py)
        with st.container(key="vitaux"):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("FC (bpm)", vitals["fc"])
            c2.metric("SpO2 (%)", vitals["spo2"])
            c3.metric("Température (°C)", vitals["temperature"])
            c4.metric("Lactate (mmol/L)", vitals["lactate"])
            c5.metric("Glucose (mg/dL)", vitals["glucose"])

        signal_ecg = generer_signal_ecg(hr_bpm=vitals["fc"], duree_s=2.5)
        temps_axe = np.linspace(0, 2.5, len(signal_ecg))
        fig_ecg = go.Figure()
        fig_ecg.add_trace(go.Scatter(x=temps_axe, y=signal_ecg, mode="lines",
                                      line=dict(color=PALETTE["accent"], width=2), name="ECG"))
        fig_ecg.update_layout(title="Tracé ECG synthétique", template=MODELE_PLOTLY,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240,
                               margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Temps (s)", yaxis_title="Amplitude (mV)")
        tracer(fig_ecg)

        if len(st.session_state.historique_vitals) > 1:
            df_hist = (pd.DataFrame(list(st.session_state.historique_vitals))
                       .rename(columns={"fc": "FC (bpm)", "spo2": "SpO2 (%)"}))
            fig_hist = px.line(df_hist, x="horodatage", y=["FC (bpm)", "SpO2 (%)"], template=MODELE_PLOTLY,
                                title="Historique récent : FC et SpO2",
                                labels={"horodatage": "Heure", "value": "", "variable": ""})
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240,
                                   margin=dict(l=20, r=20, t=40, b=20))
            tracer(fig_hist)

    @st.fragment(run_every=intervalle_rafraichissement if st.session_state.simulation_active else None)
    def fragment_monitoring():
        if st.session_state.simulation_active:
            nouvel_etat = simuler_constantes_vitales(profil_athlete, st.session_state.dernier_etat_vital)
            st.session_state.dernier_etat_vital = nouvel_etat
            st.session_state.historique_vitals.append(nouvel_etat)
            afficher_etat_vital(nouvel_etat)
        else:
            etat_affiche = st.session_state.dernier_etat_vital or simuler_constantes_vitales(profil_athlete)
            st.session_state.dernier_etat_vital = etat_affiche
            afficher_etat_vital(etat_affiche)
            st.info("Simulation en pause. Activez le mode Live pour un flux continu de données.")

    fragment_monitoring()


# ONGLET 2 : CHARGE & FATIGUE (ACWR)
with onglet2:
    st.subheader("Analyse de la charge d'entraînement (ACWR)")
    col_g1, col_g2 = st.columns([1, 2])

    with col_g1:
        fig_jauge = go.Figure(go.Indicator(
            mode="gauge+number", value=acwr_info["acwr"], title={"text": "ACWR (7j / 28j)"},
            gauge={"axis": {"range": [0, 2.2]}, "bar": {"color": acwr_info["couleur"]},
                   "steps": [{"range": [0, 0.8], "color": "#DCEAF0"}, {"range": [0.8, 1.3], "color": "#DCF0E4"},
                             {"range": [1.3, 1.5], "color": "#FBEBCF"}, {"range": [1.5, 2.2], "color": "#F9D9DE"}],
                   "threshold": {"line": {"color": PALETTE["encre"], "width": 3}, "thickness": 0.8, "value": acwr_info["acwr"]}},
        ))
        fig_jauge.update_layout(template=MODELE_PLOTLY, paper_bgcolor="rgba(0,0,0,0)", height=320,
                                 margin=dict(l=20, r=20, t=60, b=10))
        tracer(fig_jauge)
        st.markdown(f"**Zone actuelle :** {acwr_info['zone']}")
        st.caption(f"Charge aiguë (7j) : {acwr_info['aigue']}  |  Charge chronique (28j) : {acwr_info['chronique']}")

    with col_g2:
        fig_charge = px.bar(df_charge, x="date", y="charge", template=MODELE_PLOTLY,
                             title="Charge d'entraînement journalière (Session-RPE)")
        fig_charge.update_traces(marker_color="#5B9BB0")
        fig_charge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
        tracer(fig_charge)

    if acwr_info["jours_historique"] < 28:
        st.caption(f"ACWR provisoire : {acwr_info['jours_historique']} jour(s) d'historique sur les 28 de la "
                   "charge chronique.")

    st.markdown("#### Enregistrer une séance")
    _afficher_flash("seance")
    if id_reel:
        st.caption(f"La séance est écrite dans le journal réel de **{id_reel}** (onglet Données Réelles).")
    else:
        st.caption("Mode simulation : la séance est gardée pendant la session. Pour l'enregistrer sur disque, "
                   "choisis un athlète du journal dans « Données d'entraînement » (barre latérale).")
    with st.form("form_seance", clear_on_submit=True):
        s1, s2, s3 = st.columns(3)
        date_seance = s1.date_input("Date", value=aujourdhui, max_value=aujourdhui, format="DD/MM/YYYY")
        rpe_seance = s2.slider("Effort perçu (RPE, 0-10)", 0, 10, 6)
        duree_seance = s3.number_input("Durée (min)", min_value=0, max_value=300, value=60, step=5)
        st.caption("Charge de la séance = RPE × durée (méthode sRPE de Foster). Deux séances le même jour s'additionnent.")
        seance_validee = st.form_submit_button("Enregistrer la séance")
    if seance_validee:
        charge_seance = rpe_seance * duree_seance
        try:
            if duree_seance <= 0:
                raise ErreurJournal("Durée nulle : indique la durée de la séance en minutes.")
            if id_reel:
                journal_maj, _ = ajouter_seance(id_reel, date_seance, rpe_seance, duree_seance)
                acwr_apres = calculer_acwr(charge_28j(journal_maj, id_reel))
            else:
                st.session_state.seances_simulees.append(
                    {"athlete": nom_athlete, "date": date_seance, "charge": float(charge_seance)})
                acwr_apres = calculer_acwr(charge_simulee(nom_athlete))
            _flash("seance", f"Séance du {date_seance:%d/%m} enregistrée : charge {charge_seance} "
                             f"({rpe_seance} × {duree_seance} min). ACWR : {acwr_info['acwr']} → {acwr_apres['acwr']}.")
            st.rerun()
        except ErreurJournal as err:
            st.error(str(err))


# ONGLET 3 : NUTRITION & MÉTABOLISME
with onglet3:
    st.subheader("Suivi nutritionnel et métabolique")
    col_n1, col_n2, col_n3 = st.columns(3)
    col_n1.metric("BMR (métabolisme de base)", f"{bmr_tdee['bmr']} kcal")
    col_n2.metric("TDEE (besoin journalier)", f"{bmr_tdee['tdee']} kcal")
    col_n3.metric("Kcal consommées", f"{int(kcal_consommees)} kcal",
                  delta=f"{int(kcal_consommees - bmr_tdee['tdee'])} kcal vs objectif")

    st.markdown("#### Enregistrer un repas")
    st.caption(f"Repas enregistrés pour **{athlete_courant}**, le {aujourdhui:%d/%m/%Y}. Ils sont gardés sur "
               "l'ordinateur (fichier data/real/repas.csv).")
    _afficher_flash("repas")
    if erreur_repas:
        st.error(erreur_repas)
    with st.form("form_repas", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        nom_repas = c1.text_input("Repas (facultatif)", placeholder="Ex : Petit-déjeuner")
        kcal_repas = c2.number_input("Kcal", min_value=0, value=0, step=10)
        proteines = c3.number_input("Protéines (g)", min_value=0, value=0, step=1)
        glucides = c4.number_input("Glucides (g)", min_value=0, value=0, step=1)
        lipides = c5.number_input("Lipides (g)", min_value=0, value=0, step=1)
        valider = st.form_submit_button("Enregistrer le repas")
    if valider:
        try:
            ligne = ajouter_repas(athlete_courant, aujourdhui, datetime.now().strftime("%H:%M"), nom_repas,
                                  kcal_repas, proteines, glucides, lipides)
            _flash("repas", f"Repas « {ligne['repas']} » enregistré ({kcal_repas} kcal). "
                            f"Total du jour : {int(kcal_consommees + kcal_repas)} kcal.")
            st.rerun()
        except ErreurJournal as err:
            st.error(str(err))

    if len(repas_du_jour):
        st.markdown("#### Repas enregistrés aujourd'hui")
        repas_affiches = repas_du_jour.rename(columns={
            "heure": "Heure", "repas": "Repas", "kcal": "Kcal", "proteines_g": "Protéines (g)",
            "glucides_g": "Glucides (g)", "lipides_g": "Lipides (g)"})
        tableau(repas_affiches[["Heure", "Repas", "Kcal", "Protéines (g)", "Glucides (g)", "Lipides (g)"]])
        d1, d2 = st.columns([3, 1])
        a_supprimer = d1.selectbox(
            "Corriger une erreur de saisie", repas_du_jour["id"].tolist(), key="repas_a_supprimer",
            format_func=lambda i: "{heure} · {repas} · {kcal:.0f} kcal".format(
                **repas_du_jour.set_index("id").loc[i].to_dict()))
        if d2.button("Supprimer ce repas"):
            try:
                supprimer_repas(a_supprimer)
                _flash("repas", "Repas supprimé.", "info")
                st.rerun()
            except ErreurJournal as err:
                st.error(str(err))
        col_macro1, col_macro2 = st.columns(2)
        with col_macro1:
            total_macros = repas_affiches[["Protéines (g)", "Glucides (g)", "Lipides (g)"]].sum()
            fig_macros = px.pie(names=total_macros.index, values=total_macros.values, template=MODELE_PLOTLY,
                                 title="Répartition des macronutriments",
                                 color_discrete_sequence=["#1B6E8C", "#5B9BB0", "#FFB703"])
            fig_macros.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320)
            tracer(fig_macros)
        with col_macro2:
            fig_bilan = go.Figure(go.Bar(x=["Consommé", "Objectif (TDEE)"], y=[kcal_consommees, bmr_tdee["tdee"]],
                                          marker_color=["#1B6E8C", "#5B7285"]))
            fig_bilan.update_layout(template=MODELE_PLOTLY, title="Bilan calorique du jour",
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
            tracer(fig_bilan)
    else:
        st.info("Aucun repas enregistré pour le moment.")

    st.markdown("#### Panel micronutriments")
    df_micro_affiche = df_micro.copy()
    df_micro_affiche["Statut"] = df_micro_affiche.apply(
        lambda l: "Carence" if l["Valeur mesurée"] < l["Seuil minimal"] else "Normal", axis=1)
    tableau(df_micro_affiche)


# ONGLET 4 : DECISION ENGINE (règles)
with onglet4:
    st.subheader("Moteur de décision — Alertes prioritaires & recommandations")
    st.caption("Synthèse générée automatiquement à destination du staff technique et du coach.")

    nb_critiques = sum(1 for a in alertes if a[0] == "Critique")
    nb_attention = sum(1 for a in alertes if a[0] == "Attention")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Alertes critiques", nb_critiques)
    col_r2.metric("Points d'attention", nb_attention)
    col_r3.metric("ACWR actuel", acwr_info["acwr"])

    st.markdown("---")
    st.markdown("### Recommandations priorisées")
    for niveau, titre, recommandation in alertes:
        carte_alerte(niveau, titre, recommandation)


# ONGLET 5 : IA PRÉDICTIVE (Injury Risk, AI Coach, Performance Score, Rapport)
with onglet5:
    st.subheader("Intelligence artificielle prédictive")
    st.caption("Modèles Gradient Boosting (scikit-learn) — Injury Risk, Fatigue, Overtraining, Performance Score.")

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        carte_kpi(f"{profil_risque['injury_risk_pct']} / 100",
                  f"Score de risque · probabilité estimée {profil_risque['probabilite_prevalence_pct']} %",
                  profil_risque["injury_risk_categorie"]["niveau"])
    with col_k2:
        carte_kpi(f"{profil_risque['fatigue_score']} / 10", "Niveau de fatigue",
                  profil_risque["fatigue_categorie"]["niveau"])
    with col_k3:
        carte_kpi(f"{profil_risque['overtraining_pct']} %", "Risque de surentraînement",
                  profil_risque["overtraining_categorie"]["niveau"])

    st.caption(
        f"Tendance du risque sur 7 jours : {tendance_sens['texte']} "
        f"— recalculée jour par jour à partir de l'historique réel de charge (ACWR glissant)."
    )
    fig_tendance = go.Figure(go.Scatter(
        x=[t["jour_offset"] for t in tendance_trajectoire],
        y=[t["injury_risk_pct"] for t in tendance_trajectoire],
        mode="lines+markers", line=dict(color="#1B6E8C", width=3), marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(27,110,140,0.08)",
    ))
    fig_tendance.update_layout(template=MODELE_PLOTLY, height=160, paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title=None, yaxis_title="Score de risque (/100)")
    tracer(fig_tendance)

    st.markdown("---")
    col_perf1, col_perf2 = st.columns([1, 1])

    with col_perf1:
        st.markdown("#### Athlete Performance Score")
        fig_score = go.Figure(go.Indicator(
            mode="gauge+number", value=performance["score_global"],
            number={"font": {"color": PALETTE["encre"], "size": 52}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": PALETTE["accent"]},
                   "steps": [{"range": [0, 50], "color": "#F9D9DE"}, {"range": [50, 75], "color": "#FBEBCF"},
                             {"range": [75, 100], "color": "#DCF0E4"}]},
        ))
        fig_score.update_layout(template=MODELE_PLOTLY, paper_bgcolor="rgba(0,0,0,0)", height=260,
                                 margin=dict(l=20, r=20, t=20, b=10))
        tracer(fig_score)

    with col_perf2:
        st.markdown("#### Radar de performance")
        dims = performance["dimensions"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(dims.values()) + [list(dims.values())[0]],
            theta=list(dims.keys()) + [list(dims.keys())[0]],
            fill="toself", line=dict(color=PALETTE["accent"], width=2), fillcolor="rgba(27,110,140,0.18)",
        ))
        fig_radar.update_layout(template=MODELE_PLOTLY, paper_bgcolor="rgba(0,0,0,0)", height=290,
                                 polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                 margin=dict(l=30, r=30, t=20, b=20))
        tracer(fig_radar)

    st.markdown("---")
    col_f1, col_f2 = st.columns([1, 1.3])
    with col_f1:
        st.markdown("#### Fiche du modèle")
        meta = profil_risque.get("metadata_modele", {})
        if meta.get("metrics"):
            m_1, m_2 = st.columns(2)
            m_1.metric("ROC-AUC (test)", f"{meta['metrics'].get('injury_model_auc', 0):.3f}")
            m_2.metric("MAE fatigue (/10)", f"{meta['metrics'].get('fatigue_model_mae', 0):.3f}")
            profondeur = meta.get("hyperparametres", {}).get("injury_risk_model", {}).get("max_depth", "?")
            st.caption(f"Gradient Boosting (scikit-learn), profondeur {profondeur}, entraîné sur "
                       f"{meta.get('n_train_samples')} profils simulés. Le détail des facteurs propres "
                       f"à cet athlète est présenté plus bas.")
        if meta.get("avertissement_version"):
            st.warning(meta["avertissement_version"])

    with col_f2:
        st.markdown("#### AI Coach — Recommandations")
        for r in recommandations:
            niveau_reco = {1: "Critique", 2: "Attention", 3: "Info", 4: "OK"}.get(r["priorite"], "Info")
            carte_alerte(niveau_reco, r["categorie"], r["texte"])

    st.markdown("---")
    st.markdown("#### Pourquoi ce score ? (explication individuelle)")
    st.caption("Pour CET athlète : de combien chaque facteur pousse son risque vers le haut (rouge) ou le bas (vert) "
               "par rapport à un profil de référence sain, recalculé avec le modèle de production.")
    fig_expl = go.Figure(go.Bar(
        x=[e["contribution_pts"] for e in explication_individuelle][::-1],
        y=[f"{e['facteur']} ({e['valeur']})" for e in explication_individuelle][::-1],
        orientation="h",
        marker_color=[PALETTE["critique"] if e["contribution_pts"] > 0 else PALETTE["ok"] for e in explication_individuelle][::-1],
    ))
    fig_expl.update_layout(template=MODELE_PLOTLY, height=290, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=30),
                            xaxis_title="Contribution au score de risque (points)")
    tracer(fig_expl)

    st.markdown("#### Plan d'action chiffré")
    if plan_action["leviers"]:
        for lv in plan_action["leviers"]:
            st.markdown(f"- **{lv['action']}** ({lv['de']} → {lv['vers']}) : **−{lv['gain_pts']} pts** de score")
        pc = plan_action["plan_combine"]
        st.success(f"Plan combiné : score de risque {pc['avant_pct']} → **{pc['apres_pct']} / 100** (−{pc['gain_pts']} pts).")
    else:
        st.info("Aucun levier à fort impact : le profil est déjà proche de l'optimum sur les facteurs actionnables.")

    with st.expander("Simulateur What-If — testez un scénario"):
        c_w1, c_w2, c_w3 = st.columns(3)
        w_sommeil = c_w1.slider("Sommeil (h)", 3.5, 10.5, float(min(max(sommeil_h, 3.5), 10.5)), 0.1, key="w_sommeil")
        w_acwr = c_w2.slider("ACWR", 0.3, 2.5, float(min(max(acwr_info["acwr"], 0.3), 2.5)), 0.05, key="w_acwr")
        w_int = c_w3.slider("Intensité moyenne 7j", 0.05, 1.0, float(min(max(intensite_effort, 0.05), 1.0)), 0.05, key="w_int")
        res_w = simuler_scenario(features, {"sommeil_h": w_sommeil, "acwr": w_acwr, "intensite_moyenne_7j": w_int})
        m1, m2, m3 = st.columns(3)
        m1.metric("Score actuel", f"{res_w['avant_pct']} / 100")
        m2.metric("Score simulé", f"{res_w['apres_pct']} / 100", delta=f"{-res_w['gain_pts']:+.1f} pts", delta_color="inverse")
        m3.metric("Gain", f"{res_w['gain_pts']:+.1f} pts")
        st.caption("Estimation du modèle sur données synthétiques : outil d'aide à la décision, pas une garantie.")

    st.markdown("---")
    st.markdown("#### Rapport IA exportable")
    if st.button("Generate AI Report"):
        try:
            with st.spinner("Génération du rapport PDF en cours..."):
                chemin_temp = str(Path(tempfile.gettempdir()) / f"athena_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                generer_rapport_pdf(chemin_temp, profil_athlete, profil_risque, acwr_info,
                                     performance, recommandations, alertes,
                                     etude_scientifique=etude_scientifique_en_cache(),
                                     explication=explication_individuelle, plan_action=plan_action)
                with open(chemin_temp, "rb") as f:
                    donnees_pdf = f.read()
            st.download_button("Télécharger le rapport PDF", data=donnees_pdf,
                                file_name=f"ATHENA_Rapport_{athlete_selectionne.split()[0]}.pdf",
                                mime="application/pdf")
            st.success("Rapport généré avec succès.")
        except Exception as e:
            st.error(f"Échec de la génération du rapport PDF. Détail technique : {e}")


# ONGLET 6 : ANALYSE VIDÉO (Computer Vision)
with onglet6:
    st.subheader("Analyse biomécanique par vision par ordinateur")
    st.caption("Détection posturale via MediaPipe Pose, calibrée par type de mouvement — Football, Basketball, Athlétisme.")

    from computer_vision.pose_analysis import (
        MOUVEMENTS_REFERENCE, modele_disponible_localement, telecharger_modele, _charger_detecteur, nom_modele,
    )

    if not modele_disponible_localement():
        st.warning(
            "Le modèle de détection posturale n'est pas encore présent en local sur cette machine "
            "(il sera téléchargé une seule fois, ~9 Mo, à la première analyse — connexion requise). "
            "Pour une démo sans dépendre du wifi le jour J : lancer une fois "
            "`python -m computer_vision.pose_analysis --setup` avant l'événement."
        )
    else:
        try:
            _charger_detecteur()   # préchauffage (~2 s, une seule fois) : la 1re analyse devant le jury est immédiate
        except Exception:  # noqa: BLE001 - l'erreur sera affichée à la première analyse
            pass

    if not modele_disponible_localement():
        if st.button("Télécharger le modèle de détection (~9 Mo)"):
            try:
                with st.spinner("Téléchargement en cours..."):
                    telecharger_modele()
                st.success("Modèle installé. Vous pouvez analyser une image.")
                st.rerun()
            except Exception as e:
                st.error(f"Téléchargement impossible (connexion ?). Détail : {e}")

    col_choix1, col_choix2 = st.columns([1, 2])
    with col_choix1:
        mouvement_analyse = st.selectbox("Type de mouvement", list(MOUVEMENTS_REFERENCE.keys()))
    with col_choix2:
        st.caption("La zone d'angle « correcte » (genou/hanche/tronc) dépend du mouvement : un squat et une foulée de course n'ont pas la même posture idéale.")

    st.caption("Squelette : cuisse, tibia et tronc en **bleu** si l'angle est dans la plage de référence du "
               "mouvement, **orange** s'il en est proche (moins de 15°), **rouge** sinon. Les points mal vus ne "
               f"sont pas dessinés. Modèle de pose : {nom_modele() if modele_disponible_localement() else 'absent'}.")
    fichier_image = st.file_uploader("Image de l'athlète (JPG/PNG/WebP)", type=["jpg", "jpeg", "png", "webp"])

    if fichier_image is not None:
        import cv2
        from computer_vision.pose_analysis import analyser_image, dessiner_squelette, preparer_image

        octets_photo = fichier_image.getvalue()
        image_bgr = cv2.imdecode(np.frombuffer(octets_photo, np.uint8), cv2.IMREAD_COLOR)

        resultat = None
        if image_bgr is None:
            st.error("Image illisible ou fichier corrompu. Réessayez avec un JPG/PNG valide.")
        else:
            image_bgr = preparer_image(image_bgr)
            # Gardé en session : changer de mouvement ou toucher un curseur ne relance pas la détection.
            cle_photo = (hashlib.md5(octets_photo).hexdigest(), mouvement_analyse)
            cache_photo = st.session_state.get("analyse_photo")
            if cache_photo and cache_photo["cle"] == cle_photo:
                resultat = cache_photo["resultat"]
            else:
                try:
                    with st.spinner("Analyse posturale en cours..."):
                        resultat = analyser_image(image_bgr, mouvement=mouvement_analyse)
                    st.session_state["analyse_photo"] = {"cle": cle_photo, "resultat": resultat}
                except Exception as e:
                    st.error(
                        "Le modèle de détection posturale n'a pas pu être chargé (probablement pas de "
                        "connexion internet pour le premier téléchargement). Lancer "
                        f"`python -m computer_vision.pose_analysis --setup` avec le wifi puis réessayer. Détail : {e}"
                    )

        if resultat is None:
            pass  # erreur déjà affichée ; les autres onglets restent utilisables
        elif not resultat["detecte"]:
            st.warning("Aucune silhouette détectée sur cette image. Essayez une image où le corps entier est visible.")
        elif not resultat.get("fiable", True):
            image(cv2.cvtColor(dessiner_squelette(image_bgr, resultat["landmarks_bruts"]), cv2.COLOR_BGR2RGB),
                  caption="Squelette détecté (points mal vus masqués)")
            st.warning(resultat["message"])
        else:
            col_v1, col_v2 = st.columns([1.1, 1])
            with col_v1:
                image_annotee = dessiner_squelette(image_bgr, resultat["landmarks_bruts"], mouvement_analyse)
                image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), caption="Squelette détecté")
            with col_v2:
                st.metric("Angle du genou", f"{resultat['angle_genou']}°")
                st.metric("Angle de la hanche", f"{resultat['angle_hanche']}°")
                st.metric("Écart tronc/vertical", f"{resultat['angle_tronc_vertical']}°")
                st.metric("Score technique", f"{resultat['score_technique']} / 100")
                st.markdown(f"**Côté analysé :** {resultat['cote_analyse']} (meilleure visibilité)")
                st.markdown(f"**Évaluation :** {resultat['niveau']}")

                asym = resultat["asymetrie"]
                if not asym.get("mesurable"):
                    st.caption("Asymétrie gauche/droite non évaluable — un seul côté du corps est assez visible sur cette image.")
                else:
                    st.markdown(
                        f"**Asymétrie gauche/droite :** {asym['niveau']} "
                        f"(écart genou {asym['ecart_genou']}°, écart hanche {asym['ecart_hanche']}°)"
                    )
                    if asym["niveau"] == "Notable":
                        st.caption("Un écart marqué entre les deux côtés est un signal classique de compensation posturale — à recouper avec le staff technique, pas un diagnostic en soi.")
    else:
        st.caption("Aucune image chargée. En l'absence de flux vidéo, ce module accepte une image fixe pour la démonstration.")


    st.markdown("---")
    st.markdown("#### Analyse d'une séquence vidéo (5 à 15 s)")
    st.caption(
        "Au lieu d'une photo unique : angles genou/hanche/tronc image par image, amplitude, symétrie "
        "gauche/droite, cadence, et score évalué à l'image clé de chaque foulée/répétition. "
        "Filmer le corps entier, caméra fixe, bonne lumière — de profil pour genou/hanche, de face ou "
        "de dos pour la symétrie. Démonstrateur 2D, pas un outil médical certifié."
    )
    fichier_video = st.file_uploader("Vidéo de l'athlète (MP4/MOV/AVI/M4V/WebM)",
                                     type=["mp4", "mov", "avi", "m4v", "webm"], key="upload_video_seq")
    if fichier_video is not None:
        import cv2
        from computer_vision.analyse_sequence import extraire_pose_video, analyser_series, annoter_images, encoder_video
        octets_video = fichier_video.getvalue()
        # 1. Détection de la pose (10-20 s) : UNE fois par vidéo, gardée en session (clé = empreinte du
        #    fichier). Changer de mouvement ne relance que le calcul des angles et le dessin : instantané.
        empreinte_video = hashlib.md5(octets_video).hexdigest()
        cache_video = st.session_state.get("analyse_video")
        if not cache_video or cache_video["cle"] != empreinte_video:
            with tempfile.NamedTemporaryFile(suffix=Path(fichier_video.name).suffix, delete=False) as tmp_video:
                tmp_video.write(octets_video)
                chemin_tmp_video = tmp_video.name
            cache_video = {"cle": empreinte_video, "extraction": None, "originale": None, "annotees": {}, "erreur": None}
            try:
                barre = st.progress(0.0, text="Détection de la pose, image par image…")
                extraction = extraire_pose_video(
                    chemin_tmp_video, rappel_progression=lambda p: barre.progress(
                        p, text=f"Détection de la pose, image par image… {p:.0%}"))
                barre.progress(1.0, text="Préparation des vidéos…")
                cache_video["extraction"] = extraction
                # Vidéo d'origine réencodée (640 px, WebM) : un MOV HEVC d'iPhone devient lisible partout.
                cache_video["originale"] = encoder_video(extraction["images"], extraction["fps"])
                barre.empty()
            except ValueError as e:
                cache_video["erreur"] = f"Vidéo non exploitable : {e}"
            except Exception as e:  # noqa: BLE001 - modèle de pose absent, etc.
                cache_video["erreur"] = ("L'analyse n'a pas pu s'exécuter (modèle de pose absent ? lancer "
                                         f"`python -m computer_vision.pose_analysis --setup` avec internet). Détail : {e}")
            finally:
                Path(chemin_tmp_video).unlink(missing_ok=True)
            st.session_state["analyse_video"] = cache_video

        # 2. Angles, score et squelette annoté pour le mouvement choisi (rapide, gardé par mouvement).
        extraction, resultat_seq, annotees = cache_video["extraction"], None, None
        if extraction is not None:
            resultat_seq = analyser_series(extraction["serie"], extraction["fps"], mouvement_analyse)
            resultat_seq["fps"] = round(extraction["fps"], 1)
            if mouvement_analyse not in cache_video["annotees"]:
                images_annotees = annoter_images(extraction["images"], extraction["landmarks"], mouvement_analyse)
                cache_video["annotees"][mouvement_analyse] = {
                    "images": images_annotees, "video": encoder_video(images_annotees, extraction["fps"])}
            annotees = cache_video["annotees"][mouvement_analyse]
            duree_totale = extraction.get("duree_video_s") or 0
            if duree_totale > 15.5:
                st.info(f"Vidéo de {duree_totale:.0f} s : seules les 15 premières secondes sont analysées. "
                        "Couper la vidéo autour du geste à analyser.")

        col_vid1, col_vid2 = st.columns(2)
        with col_vid1:
            st.markdown("**Vidéo d'origine**")
            if cache_video["originale"]:
                st.video(cache_video["originale"][0], format=cache_video["originale"][1])
            else:
                st.video(octets_video)
        with col_vid2:
            st.markdown("**Squelette détecté (images analysées)**")
            if annotees and annotees["video"]:
                st.video(annotees["video"][0], format=annotees["video"][1], loop=True, autoplay=True, muted=True)
            elif annotees:
                st.caption("Vidéo annotée indisponible (aucun encodeur vidéo sur cette machine) : "
                           "le curseur ci-dessous montre chaque image analysée.")
        if annotees:
            # Toujours disponible : arrêter sur une image précise pour la commenter devant le jury.
            n_images = len(annotees["images"])
            c_img, c_txt = st.columns([1, 1])
            with c_img:
                num_image = st.slider("Parcourir les images analysées", 1, n_images, 1, key=f"image_{empreinte_video}")
                t_image = extraction["serie"]["t"].iloc[num_image - 1]
                image(cv2.cvtColor(annotees["images"][num_image - 1], cv2.COLOR_BGR2RGB),
                      caption=f"Image {num_image}/{n_images} · t = {t_image:.1f} s")
            with c_txt:
                n_detectees = sum(lm is not None for lm in extraction["landmarks"])
                st.caption(f"Modèle de pose : {extraction['modele']}, en mode vidéo (la personne est suivie "
                           f"d'une image à l'autre). Silhouette détectée sur {n_detectees} images sur {n_images}.")
        if cache_video["erreur"]:
            st.error(cache_video["erreur"])

        if resultat_seq is not None:
            if resultat_seq["statut"] != "ok":
                st.warning(resultat_seq["message"])
            else:
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Score technique", f"{resultat_seq['score_technique']} / 100")
                q2.metric("Indice de symétrie", f"{resultat_seq['indice_symetrie']} %" if resultat_seq["indice_symetrie"] is not None else "n/d")
                q3.metric("Cadence", f"{int(resultat_seq['cadence_pas_min'])} pas/min" if resultat_seq["cadence_pas_min"] else "n/d")
                q4.metric("Images détectées", f"{int(resultat_seq['taux_detection'] * 100)} %")
                st.markdown(f"**Évaluation :** {resultat_seq['niveau']}")
                for ligne_expl in resultat_seq["explications"]:
                    st.markdown(f"- {ligne_expl}")
                serie_aff = resultat_seq["serie_lissee"]
                fig_seq = go.Figure()
                fig_seq.add_trace(go.Scatter(x=serie_aff["t"], y=serie_aff["genou_g"], name="Genou gauche", line=dict(color="#1B6E8C")))
                fig_seq.add_trace(go.Scatter(x=serie_aff["t"], y=serie_aff["genou_d"], name="Genou droit", line=dict(color="#E67E22")))
                fig_seq.update_layout(template=MODELE_PLOTLY, height=320, margin=dict(l=20, r=20, t=40, b=20),
                                      title="Angle du genou au cours du temps", xaxis_title="Temps (s)", yaxis_title="Angle (°)")
                tracer(fig_seq)
                st.caption(f"{resultat_seq['n_images']} images analysées sur {resultat_seq['duree_s']} s "
                           f"({resultat_seq['fps']} images/s). Aucune vidéo n'est conservée sur le disque : "
                           "elle reste en mémoire le temps de la session.")

# ONGLET 7 : NATIONAL TEAM DASHBOARD (multi-athlètes)
with onglet7:
    st.subheader("National Team Dashboard — Vue centre de performance")
    st.caption("Surveillance simultanée de l'ensemble de l'effectif — classement, risque, fatigue, disponibilité.")

    df_equipe = generer_snapshot_equipe()

    lignes_scored = []
    for _, ligne in df_equipe.iterrows():
        athlete_ref = next(a for a in EQUIPE_NATIONALE_DEMO if a["nom"] == ligne["Nom"])
        if ligne["Nom"] == nom_athlete:
            # Athlète sélectionné : mêmes données que le bandeau et l'onglet IA Prédictive
            # (barre latérale + charge réelle), pour qu'un même athlète n'ait qu'un seul score.
            profil, acwr_ligne, nom_ligne = profil_risque, acwr_info["acwr"], f"{ligne['Nom']} (sélection)"
        else:
            feats = construire_vecteur_features(
                acwr=ligne["ACWR"], charge_chronique=420, fc_repos=60, sommeil_h=ligne["Sommeil (h)"],
                age=ligne["Âge"], imc=athlete_ref["poids"] / (athlete_ref["taille"] / 100) ** 2,
                historique_blessures=ligne["Historique blessures"],
                deficit_calorique_pct=0, lactate=2.5, intensite_moyenne_7j=athlete_ref["intensite_effort"],
            )
            profil, acwr_ligne, nom_ligne = predire_profil_risque(feats), ligne["ACWR"], ligne["Nom"]
        disponibilite = "Disponible" if profil["injury_risk_pct"] < 50 else "À surveiller"
        lignes_scored.append({
            "Athlète": nom_ligne, "Sport": ligne["Sport"], "ACWR": acwr_ligne,
            "Score de risque (/100)": profil["injury_risk_pct"],
            "Probabilité estimée (%)": profil["probabilite_prevalence_pct"], "Fatigue (/10)": profil["fatigue_score"],
            "Overtraining (%)": profil["overtraining_pct"], "Disponibilité": disponibilite,
        })

    df_classement = pd.DataFrame(lignes_scored).sort_values("Score de risque (/100)")

    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("Effectif suivi", len(df_classement))
    col_e2.metric("Athlètes à surveiller", (df_classement["Disponibilité"] == "À surveiller").sum())
    col_e3.metric("Score de risque moyen", f"{df_classement['Score de risque (/100)'].mean():.1f} / 100")

    st.markdown("#### Classement par risque (du plus sûr au plus exposé)")
    _pleine_largeur(
        st.dataframe,
        df_classement.style.background_gradient(subset=["Score de risque (/100)"], cmap="RdYlGn_r", vmin=0, vmax=100)
        .format({"ACWR": "{:.2f}", "Score de risque (/100)": "{:.1f}", "Probabilité estimée (%)": "{:.1f}",
                 "Fatigue (/10)": "{:.1f}", "Overtraining (%)": "{:.1f}"}),
        hide_index=True,
    )

    fig_equipe = px.bar(df_classement, x="Athlète", y=["Score de risque (/100)", "Overtraining (%)"],
                         barmode="group", template=MODELE_PLOTLY, title="Risque de blessure vs surentraînement par athlète",
                         labels={"value": "/ 100", "variable": ""})
    fig_equipe.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
    tracer(fig_equipe)


# ONGLET 8 : ÉTUDE SCIENTIFIQUE (Research Question / Baseline / Results / Discussion)
with onglet8:
    st.subheader("Étude scientifique")
    st.caption("Même contenu et mêmes chiffres que le rapport final : résultat principal (E2, E3) "
               "d'abord, puis l'expérience de départ (E1).")

    e2, e3 = experiences_e2_e3_en_cache()
    if e2 and e3:
        p2 = e2["P2_jeux_independants"]
        st.markdown("#### Question de recherche (E2)")
        st.info("Comment la profondeur maximale d des arbres, c'est-à-dire l'ordre d'interaction qu'un "
                "Gradient Boosting peut représenter, module-t-elle son AUC en généralisation, et quelle "
                "part de l'écart à l'AUC maximale possible est due au modèle plutôt qu'au bruit des données ?")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("AUC, GB profondeur 1", f"{p2['gb_fixe']['1']['moyenne']:.3f}",
                  help="Moyenne sur 10 jeux générés indépendamment (protocole P2).")
        k2.metric("Oracle (plafond)", f"{p2['oracle']['moyenne']:.3f}",
                  help="AUC du risque latent sans bruit : le meilleur AUC atteignable avec ces 10 variables.")
        eo = p2["ecart_oracle"]["gb_fixe"]
        k3.metric("Écart à l'oracle", f"{eo['moyenne']:.3f}",
                  help=f"IC95 [{eo['ic95'][0]:.3f} ; {eo['ic95'][1]:.3f}]")
        dl = p2["diff_profondeur1_vs_lr"]["gb_fixe"]
        k4.metric("Gain vs régression logistique", f"{dl['moyenne']:+.3f}",
                  help=f"IC95 [{dl['ic95'][0]:.3f} ; {dl['ic95'][1]:.3f}]")
        st.caption("Le plafond d'environ 0,70 vient du bruit non observé du générateur, pas du modèle : "
                   "le GB de profondeur 1 n'est qu'à 0,007 de l'oracle. Intervalles de confiance au survol.")

        st.markdown("#### Hypothèses pré-enregistrées et verdicts")
        libelles = {
            "H2": "Ajouter des interactions (profondeur d > 1) n'améliore pas l'AUC sur un risque additif",
            "H3": "L'écart entre la meilleure profondeur et l'oracle est inférieur à 0,030",
            "H4": "La profondeur 1 bat la régression logistique (borne basse de l'IC95 > 0)",
            "H5": "Une interaction modérée (c = 0,5) est détectée par la profondeur 2",
            "H6": "La profondeur 3 n'apporte rien de plus que la profondeur 2",
            "H7": "La profondeur 2 minimise le pire regret quand la structure des données est inconnue",
            "H8": "Réplication de E2 : sans interaction, la profondeur 2 perd face à la profondeur 1",
        }
        verdicts = {**e2.get("verdicts", {}), **e3.get("verdicts", {})}
        for code_h, libelle in libelles.items():
            verdict = verdicts.get(code_h, {}).get("verdict", "n/d")
            classe = "pastille-ok" if verdict == "confirmée" else "pastille-attention"
            st.markdown(
                f'<div class="carte-alerte" style="--alerte-couleur:'
                f'{PALETTE["ok"] if verdict == "confirmée" else PALETTE["attention"]};padding:8px 14px;">'
                f'<span class="{classe}">{verdict}</span>'
                f'<strong style="margin-left:10px;">{code_h}</strong> · {libelle}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### E2 : effet de la profondeur des arbres")
        lignes_e2 = []
        for d in ["1", "2", "3", "4", "6"]:
            diff = p2["diff_vs_profondeur1"]["gb_fixe"].get(d)
            lignes_e2.append({
                "Profondeur d": int(d),
                "AUC (180 arbres)": f"{p2['gb_fixe'][d]['moyenne']:.3f} ± {p2['gb_fixe'][d]['ecart_type']:.3f}",
                "ΔAUC vs d = 1 [IC95]": "référence" if diff is None else
                    f"{diff['moyenne']:+.3f} [{diff['ic95'][0]:+.3f} ; {diff['ic95'][1]:+.3f}]",
                "AUC (arrêt précoce)": f"{p2['gb_arret'][d]['moyenne']:.3f}",
                "Arbres retenus": round(p2["n_arbres_arret"][d]),
            })
        tableau(pd.DataFrame(lignes_e2).set_index("Profondeur d"))
        fig_e2 = DOSSIER_FIGURES_RAPPORT / "fig4_profondeur.png"
        if fig_e2.exists():
            image(str(fig_e2))
        st.caption("Chaque niveau d'interaction supplémentaire retire de l'AUC : le modèle paie la variance "
                   "d'interactions qui n'existent pas dans les données. C'est ce qui expliquait le résultat "
                   "nul de E1, qui utilisait d = 3.")

        st.markdown("#### E3 : et si les données contiennent une interaction ?")
        fig_e3 = DOSSIER_FIGURES_RAPPORT / "fig7_interaction.png"
        if fig_e3.exists():
            image(str(fig_e3))
        h7 = e3["verdicts"]["H7"]
        st.caption(f"Avec une interaction forte, la profondeur 1 perd jusqu'à {h7['regret_max_d1']:.3f} d'AUC ; "
                   f"la profondeur 2 ne perd jamais plus de {h7['regret_max_d2']:.3f}. C'est le choix qui "
                   "minimise le pire regret, et la profondeur retenue en production.")
        st.markdown("---")

    st.markdown("### E1 : familles de modèles (point de départ)")
    exp = etude_scientifique_en_cache()

    st.markdown("#### Référence scientifique")
    st.markdown(f"*{exp['reference_scientifique']}*")
    st.caption(
        "Cette revue de 38 études conclut que les modèles à base d'arbres (Random Forest, "
        "XGBoost/Gradient Boosting) obtiennent la meilleure performance de prédiction (AUC) "
        "dans la majorité des études, devant la régression logistique — c'est cette observation "
        "de la littérature qu'ATHÉNA teste ici sur son propre cas d'usage."
    )

    st.markdown("#### Research Question")
    st.info(exp["research_question"])

    st.markdown("#### Hypothèse")
    col_h1, col_h0 = st.columns(2)
    col_h1.success(f"**H1** : {exp['hypothese_h1']}")
    col_h0.warning(f"**H0** : {exp['hypothese_h0']}")

    st.markdown("#### Conditions expérimentales")
    cond = exp["conditions_experimentales"]
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    col_c1.metric("Échantillons", cond["n_echantillons"])
    col_c2.metric("Train / Test", f"{cond['n_train']} / {cond['n_test']}")
    col_c3.metric("Split", cond["split"])
    col_c4.metric("Features", len(cond["features"]))
    st.caption(f"Pondération : {cond['ponderation']}, pour une comparaison équitable entre les modèles.")

    st.markdown("---")
    st.markdown("#### Results — Classification (Injury Risk)")
    df_clf = (pd.DataFrame(exp["classification_injury_risk"]).set_index("modele")
              .rename(columns={"accuracy": "Accuracy", "precision": "Précision", "recall": "Rappel",
                               "f1": "F1", "roc_auc": "ROC-AUC"}))
    df_clf.index.name = "Modèle"
    tableau(df_clf.style.format("{:.3f}"))
    st.caption("Pas de code couleur : les écarts entre modèles (≈ 0,02) sont du même ordre que la dispersion "
               "entre partitions (σ ≈ 0,02). Voir la validation croisée ci-dessous.")

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        image(exp["figures"]["roc_curves"])
    with col_fig2:
        image(exp["figures"]["confusion_matrix"])
    image(exp["figures"]["feature_importance"])

    st.markdown("#### Results — Régression (Fatigue Score)")
    df_reg = (pd.DataFrame(exp["regression_fatigue"]).set_index("modele")
              .rename(columns={"mae": "MAE", "rmse": "RMSE", "r2": "R²"}))
    df_reg.index.name = "Modèle"
    tableau(df_reg.style.format("{:.3f}"))

    st.markdown("---")
    st.markdown("#### Test d'hypothèse (bootstrap apparié, 1000 tirages)")
    th = exp["test_hypothese"]
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("P(AUC GB > AUC LR)", f"{th['proportion_gb_superieur'] * 100:.1f} %")
    col_t2.metric("Écart moyen AUC", f"{th['diff_auc_moyenne']:+.3f}")
    col_t3.metric("IC 95%", f"[{th['ic95_diff_auc'][0]:+.3f}, {th['ic95_diff_auc'][1]:+.3f}]")
    st.caption(
        "Sur un bootstrap apparié, le Gradient Boosting bat la régression logistique dans "
        f"{th['proportion_gb_superieur'] * 100:.0f}% des tirages — une évidence en faveur de H1, "
        "mais l'intervalle de confiance à 95% de l'écart traverse encore zéro : le gain est "
        "réel mais pas définitivement établi statistiquement sur ce volume de données."
    )

    st.markdown("#### Robustesse — Validation croisée stratifiée (5-fold)")
    st.caption(
        "Le split 80/20 unique ci-dessus peut être chanceux. Voici les 3 modèles réentraînés "
        "sur 5 splits indépendants (moyenne ± écart-type) — c'est la version robuste des résultats."
    )
    df_cv = pd.DataFrame(exp["validation_croisee_5fold"])
    df_cv["ROC-AUC"] = df_cv.apply(lambda r: f"{r['roc_auc_moyen']:.3f} ± {r['roc_auc_ecart_type']:.3f}", axis=1)
    df_cv["F1"] = df_cv.apply(lambda r: f"{r['f1_moyen']:.3f} ± {r['f1_ecart_type']:.3f}", axis=1)
    tableau(df_cv[["modele", "ROC-AUC", "F1"]].set_index("modele"))

    if exp.get("seuil_optimise"):
        s = exp["seuil_optimise"]
        st.markdown("#### Seuil de décision optimisé (F1-max)")
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        col_s1.metric("Seuil optimal", f"{s['seuil_optimal']:.3f}")
        col_s2.metric("Accuracy", f"{s['accuracy']:.3f}")
        col_s3.metric("Precision", f"{s['precision']:.3f}")
        col_s4.metric("Recall", f"{s['recall']:.3f}")
        col_s5.metric("F1", f"{s['f1']:.3f}")
        st.caption(
            "Au lieu du seuil par défaut (0.5), on choisit celui qui maximise le F1 sur le test "
            "set — pratique standard en classification déséquilibrée, distincte du réglage des "
            "poids. Le dashboard, lui, affiche un pourcentage continu par tranches (faible/modéré/"
            "élevé) plutôt qu'une décision binaire — plus informatif pour un coach qu'un seuil unique."
        )

    st.markdown("#### Discussion")
    concl = exp["conclusion"]
    plancher = e2.get("plancher_bruit_fatigue", {}) if e2 else {}
    texte_plancher = (f" Le plancher de bruit est {plancher['mae']:.3f} : le Gradient Boosting en est à "
                      f"{concl['mae_meilleur'] - plancher['mae']:.3f}, la fatigue est essentiellement résolue sur ce jeu."
                      if plancher else "")
    st.markdown(
        f"""
- **E1 n'était pas concluante** : le Gradient Boosting de profondeur 3 ne se distingue pas de la
  régression logistique (IC95 de la différence d'AUC contenant 0). Plutôt que d'en rester là, nous avons
  cherché *pourquoi* : c'est l'objet de E2.
- **E2 l'explique** : le risque du générateur est additif. Un arbre de profondeur 3 modélise des
  interactions qui n'existent pas et perd l'avantage de la non-linéarité. En profondeur 1, le Gradient
  Boosting bat nettement la régression logistique et s'approche de l'oracle à 0,007 près.
- **Le levier n'est ni le volume de données ni la famille d'algorithmes**, mais l'adéquation entre la
  structure du modèle (profondeur) et celle des données, mesurée par rapport à un plafond (l'oracle).
- **Choix de production** : la profondeur 2, plus robuste si les données réelles contiennent des
  interactions (E3, critère du pire regret).
- **Fatigue** : le Gradient Boosting réduit le MAE de
  {round((exp['regression_fatigue'][0]['mae'] - concl['mae_meilleur']) / exp['regression_fatigue'][0]['mae'] * 100, 1)} %
  par rapport à la baseline linéaire.{texte_plancher}
- **Limites** : données synthétiques, aucune performance clinique revendiquée. La prochaine étape est la
  validation sur des données réelles consenties (onglet Données Réelles).
"""
    )
    st.caption(
        "Reproductible : `python -m models.experiment_comparaison` (E1), "
        "`python -m models.experiment_profondeur` (E2), `python -m models.experiment_interaction` (E3)."
    )


# ONGLET 8 (suite) : CALIBRATION ET SOUS-GROUPES (analyse avancée, données simulées inédites)
with onglet8:
    import json as _json
    from models.analyse_avancee import RESULTATS_AVANCES_PATH, FIGURES_DIR as _FIG_DIR
    if RESULTATS_AVANCES_PATH.exists():
        avance = _json.loads(RESULTATS_AVANCES_PATH.read_text(encoding="utf-8"))
        if avance.get("figure_calibration"):
            st.markdown("---")
            st.markdown("#### Calibration — les probabilités affichées sont-elles fiables ?")
            col_cal1, col_cal2 = st.columns([1, 1])
            with col_cal1:
                image(str(_FIG_DIR / avance["figure_calibration"]))
            with col_cal2:
                cal = avance["calibration_production"]
                st.metric("Brier brut", cal["brier_score_brut"])
                st.metric("Brier après correction de prévalence", cal["brier_apres_correction_prevalence"])
                st.caption(
                    "Le score brut surestime le risque (le modèle est entraîné avec des poids qui favorisent "
                    "le rappel) ; la correction de prévalence rapproche la courbe de la diagonale. "
                    "Un écart subsiste dans les probabilités les plus hautes : d'où l'usage du score "
                    "comme aide à la décision par tranches, pas comme probabilité exacte."
                )
        if avance.get("sous_groupes"):
            st.markdown("#### Performance par sous-groupe (IC95 bootstrap)")
            lignes_sg = []
            for critere, groupes in avance["sous_groupes"].items():
                for g in groupes:
                    lignes_sg.append({
                        "Critère": critere, "Groupe": g["groupe"], "n": g["n"], "Blessures": g["n_positifs"],
                        "AUC": g["auc"] if g["fiable"] else "—",
                        "IC95": f"[{g['ic95'][0]}, {g['ic95'][1]}]" if g["fiable"] else "effectif insuffisant",
                    })
            tableau(pd.DataFrame(lignes_sg).set_index(["Critère", "Groupe"]))
            st.caption(
                "Calculé sur des données **simulées** : cela montre que le modèle ne se comporte pas "
                "très différemment d'un groupe à l'autre dans la simulation, pas qu'il est équitable "
                "en conditions réelles. À refaire sur les données réelles dès que l'effectif le permet."
            )


# ONGLET 9 : DONNÉES RÉELLES (journal d'entraînement + validation hors simulation)
with onglet9:
    st.subheader("Données réelles — du simulateur au terrain")
    st.caption(
        "Journal quotidien (30 s/jour) saisi par de vrais athlètes. Les features du modèle sont "
        "calculées à partir de ces mesures (charge = RPE × durée, méthode sRPE de Foster). "
        "Utiliser des identifiants pseudonymisés (A01, A02…) et obtenir le consentement de chaque athlète."
    )
    journal = journal_global   # chargé une seule fois, dans la barre latérale
    if erreur_journal:
        st.error(erreur_journal)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Athlètes suivis", journal["athlete_id"].nunique() if len(journal) else 0)
    col_b.metric("Jours saisis", len(journal))
    col_c.metric("Blessures déclarées", int(journal["blessure"].fillna(0).sum()) if len(journal) else 0)

    st.markdown("#### 1. Saisir une journée")
    st.caption("Une journée par athlète et par date : enregistrer la même date une 2e fois REMPLACE la journée. "
               "Pour ajouter une 2e séance le même jour, utilise l'onglet Charge & ACWR.")
    _afficher_flash("journal")
    with st.form("form_journal", clear_on_submit=False):
        f1, f2, f3 = st.columns(3)
        id_saisie = f1.text_input("Identifiant (ex. A01)", value=st.session_state.get("athlete_reel") or "A01")
        date_saisie = f2.date_input("Date", value=aujourdhui, max_value=aujourdhui, format="DD/MM/YYYY")
        rpe_saisie = f3.slider("Effort perçu de la séance (0 = repos)", 0, 10, 5)
        g1, g2, g3 = st.columns(3)
        duree_saisie = g1.number_input("Durée séance (min)", 0, 300, 60)
        sommeil_saisie = g2.number_input("Sommeil (h)", 2.0, 14.0, 7.5, step=0.25)
        fc_saisie = g3.number_input("FC au réveil (bpm)", 30, 120, 60)
        h1, h2, h3 = st.columns(3)
        # value=None : un poids non renseigné reste vide au lieu d'enregistrer 70 kg par défaut.
        poids_saisie = h1.number_input("Poids (kg, optionnel)", 30.0, 200.0, value=None, step=0.1,
                                       placeholder="Non mesuré")
        douleur_saisie = h2.slider("Douleur / courbatures (0-10)", 0, 10, 0)
        blessure_saisie = h3.checkbox("Blessure survenue ce jour")
        journee_validee = st.form_submit_button("Enregistrer la journée")
    if journee_validee:
        try:
            _, statut = ajouter_entree({
                "athlete_id": id_saisie, "date": date_saisie, "rpe": rpe_saisie,
                "duree_min": duree_saisie, "sommeil_h": sommeil_saisie, "fc_repos": fc_saisie,
                "poids_kg": poids_saisie, "douleur": douleur_saisie, "blessure": int(blessure_saisie),
            })
            id_enregistre = id_saisie.strip().upper()
            st.session_state["athlete_reel"] = id_enregistre   # l'affichage ci-dessous suit l'athlète saisi
            if statut == "remplacee":
                _flash("journal", f"Journée du {date_saisie:%d/%m/%Y} de {id_enregistre} remplacée "
                                  "(elle existait déjà).", "warning")
            else:
                _flash("journal", f"Journée du {date_saisie:%d/%m/%Y} enregistrée pour {id_enregistre}.")
            st.rerun()
        except ErreurJournal as err:
            st.error(str(err))

    st.markdown("#### 2. Importer / exporter")
    i1, i2 = st.columns(2)
    with i1:
        fichier = st.file_uploader("Importer un journal CSV", type="csv")
        if fichier is not None and st.button("Fusionner ce fichier"):
            try:
                _, rap_import = importer_csv(fichier)
            except (ValueError, ErreurJournal) as err:
                rap_import = {"statut": "illisible", "message": str(err)}
            # Le message est gardé en session : st.rerun() effacerait sinon tout affichage immédiat.
            st.session_state["rapport_import_journal"] = rap_import
            if rap_import.get("statut") == "ok":
                st.rerun()
        rap = st.session_state.get("rapport_import_journal")
        if rap:
            if rap.get("statut") == "ok":
                st.success(f"{rap['lignes_valides']} ligne(s) importée(s) sur {rap['lignes_recues']} "
                           f"({rap['valeurs_hors_plage']} valeur(s) hors plage ignorée(s), "
                           f"{rap['doublons_supprimes']} doublon(s)). Journal : {rap['lignes_totales_journal']} lignes.")
                for avert in rap.get("avertissements", []):
                    st.warning(avert)
            else:
                st.error(rap["message"])
                if rap.get("colonnes_fichier"):
                    st.caption("Colonnes du fichier : " + ", ".join(rap["colonnes_fichier"][:20])
                               + (" …" if len(rap["colonnes_fichier"]) > 20 else ""))
    with i2:
        st.download_button("Gabarit CSV vide", gabarit_csv(), "gabarit_journal_athena.csv", "text/csv")
        if len(journal):
            st.download_button("Exporter le journal (sauvegarde)", journal.to_csv(index=False),
                               "journal_athena.csv", "text/csv")
        with st.expander("Signification des colonnes"):
            for col, desc in DESCRIPTION_COLONNES.items():
                st.markdown(f"- `{col}` — {desc}")
    st.caption("Sur un hébergement gratuit le disque peut être réinitialisé : exporte régulièrement le journal.")

    if len(journal) == 0:
        st.info("Aucune donnée réelle pour l'instant. Saisis ta première journée ci-dessus, ou importe un CSV.")
    else:
        st.markdown("#### 3. Risque calculé sur données réelles")
        if st.session_state.get("athlete_reel") not in ids_journal:
            st.session_state.pop("athlete_reel", None)
        choix_athlete = st.selectbox("Athlète", ids_journal, key="athlete_reel")
        profil_reel = {}
        p1, p2, p3 = st.columns(3)
        age_reel = p1.number_input("Âge (optionnel)", 0, 60, 0)
        taille_reelle = p2.number_input("Taille cm (optionnel)", 0, 230, 0)
        if age_reel > 0:
            profil_reel["age"] = age_reel
        if taille_reelle > 0:
            profil_reel["taille_cm"] = taille_reelle
        res = risque_actuel(journal, choix_athlete, profil_reel)
        if res is None:
            st.warning("Pas de données pour cet athlète.")
        elif res["statut"] != "ok":
            st.warning(f"Historique insuffisant pour le modèle : {res['jours']} jour(s) d'historique, "
                       f"{res['jours_requis']} requis pour calculer des features fiables.")
            # En attendant, la charge et l'ACWR provisoire sont déjà visibles.
            charge_provisoire = charge_28j(journal, choix_athlete)
            acwr_provisoire = calculer_acwr(charge_provisoire)
            st.metric("ACWR provisoire", acwr_provisoire["acwr"],
                      help=f"Calculé sur {acwr_provisoire['jours_historique']} jour(s) au lieu de 28.")
            fig_prov = px.bar(charge_provisoire, x="date", y="charge", template=MODELE_PLOTLY,
                              title="Charge d'entraînement quotidienne (données réelles)")
            fig_prov.update_traces(marker_color="#1B6E8C")
            fig_prov.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            tracer(fig_prov)
        else:
            pr = res["profil"]
            r1, r2, r3 = st.columns(3)
            r1.metric("Score de risque", f"{pr['injury_risk_pct']} / 100",
                      f"probabilité estimée {pr['probabilite_prevalence_pct']} %", delta_color="off")
            r2.metric("Fatigue estimée", f"{pr['fatigue_score']}/10")
            r3.metric("Assiduité (28 j)", f"{int(res['info']['taux_saisie_28j'] * 100)} %")
            if res["info"]["non_mesures"]:
                st.caption("Features non mesurées par le journal (valeur de référence utilisée) : "
                           + ", ".join(res["info"]["non_mesures"]))
            serie = res["serie"]
            fig_reel = go.Figure()
            fig_reel.add_trace(go.Bar(x=serie.index, y=serie["charge"], name="Charge (RPE × min)",
                                      marker_color="#1B6E8C"))
            jours_bl = serie[serie["blessure"] == 1]
            if len(jours_bl):
                fig_reel.add_trace(go.Scatter(x=jours_bl.index, y=jours_bl["charge"], mode="markers",
                                              marker=dict(color="#C0392B", size=12, symbol="x"), name="Blessure"))
            fig_reel.update_layout(template=MODELE_PLOTLY, height=300, margin=dict(l=20, r=20, t=30, b=20),
                                   title="Charge d'entraînement quotidienne (données réelles)")
            tracer(fig_reel)

        st.markdown("#### 4. Le modèle marche-t-il sur de vraies données ?")
        if st.button("Lancer l'évaluation sur données réelles"):
            with st.spinner("Calcul en cours…"):
                ev = evaluer_sur_donnees_reelles(journal)
            if ev["statut"] != "ok":
                st.info(ev["message"])
            else:
                e1, e2, e3 = st.columns(3)
                e1.metric("AUC modèle ATHÉNA", ev["auc_modele"])
                e2.metric("AUC ACWR seul", ev["auc_acwr_seul"])
                e3.metric("AUC sommeil seul", ev["auc_sommeil_seul"])
                st.write(f"IC95 (bootstrap par athlète) de l'AUC du modèle : "
                         f"**{ev['ic95_auc'][0]} – {ev['ic95_auc'][1]}** — "
                         f"{ev['n_athletes']} athlètes, {ev['n_positifs']} fenêtres avec blessure, "
                         f"{ev['n_negatifs']} sans.")
                st.caption(ev["lecture"])


    st.markdown("#### 5. Évaluation sur un jeu de données public")
    st.caption("Importer d'abord le jeu avec `python -m data.importer_public` (voir README). "
               "L'évaluation exige des données quotidiennes et affiche AUC, IC95 et baselines.")
    fichiers_publics = sorted(DOSSIER_REEL.glob("public_*.csv"))
    if not fichiers_publics:
        st.info("Aucun jeu public importé pour l'instant.")
    else:
        choix_public = st.selectbox("Jeu public", [f.name for f in fichiers_publics])
        if st.button("Évaluer le jeu public"):
            df_public = pd.read_csv(DOSSIER_REEL / choix_public, parse_dates=["date"])
            with st.spinner("Calcul en cours…"):
                ev_pub = evaluer_sur_donnees_reelles(df_public)
            if ev_pub["statut"] != "ok":
                st.info(ev_pub["message"])
            else:
                u1, u2, u3 = st.columns(3)
                u1.metric("AUC modèle ATHÉNA", ev_pub["auc_modele"])
                u2.metric("AUC ACWR seul", ev_pub["auc_acwr_seul"])
                u3.metric("AUC sommeil seul", ev_pub["auc_sommeil_seul"])
                st.write(f"IC95 : **{ev_pub['ic95_auc'][0]} – {ev_pub['ic95_auc'][1]}** — "
                         f"{ev_pub['n_athletes']} athlètes, {ev_pub['n_positifs']} fenêtres avec blessure.")
                st.caption(ev_pub["lecture"])

# ONGLET 10 : DÉMO GUIDÉE (scénario fictif déterministe, 100 % hors-ligne)
with onglet10:
    from analytics.scenario_demo import construire_scenario, generer_pdf_demo, SOMMEIL_H as _SOMMEIL_DEMO
    st.subheader("Démo guidée — de l'alerte au plan d'action")
    st.caption("Scénario FICTIF et déterministe (même déroulé à chaque fois, sans internet). Les entrées sont "
               "fixées ; risque, facteurs et gains sont calculés par les vrais modèles d'ATHÉNA.")
    if "etape_demo" not in st.session_state:
        st.session_state.etape_demo = 1
    sc = construire_scenario()
    etapes = ["1. Alerte", "2. Pourquoi ?", "3. Et si… ?", "4. Plan d'action", "5. Rapport"]
    st.progress(st.session_state.etape_demo / len(etapes), text=etapes[st.session_state.etape_demo - 1])

    e = st.session_state.etape_demo
    if e == 1:
        st.markdown(f"**{sc['athlete']['nom']}** — {sc['athlete']['age']} ans, trois semaines de charge modérée "
                    "puis une montée brutale la semaine avant la compétition.")
        d1, d0, d2, d3 = st.columns(4)
        niveau_demo = sc["profil_risque"]["injury_risk_categorie"]["niveau"]
        d1.metric("Score de risque", f"{sc['profil_risque']['injury_risk_pct']} / 100", niveau_demo,
                  delta_color="normal" if niveau_demo == "Faible" else "inverse")
        d0.metric("Probabilité estimée", f"{sc['profil_risque']['probabilite_prevalence_pct']} %",
                  help="Score ramené à la fréquence réelle des blessures (≈ 20 %) par la règle de Bayes. "
                       "Vérifié sur 20 000 profils inédits : les probabilités annoncées correspondent aux fréquences observées.")
        d2.metric("ACWR", float(sc["acwr_info"]["acwr"]), sc["acwr_info"]["zone"], delta_color="off")
        d3.metric("Fatigue", f"{sc['profil_risque']['fatigue_score']} / 10")
        for niveau, titre, action in sc["alertes"][:3]:
            (st.error if niveau == "Critique" else st.warning)(f"**{titre}** — {action}")
    elif e == 2:
        st.markdown("**Qu'est-ce qui pèse sur son risque ?** (points de score vs une référence saine)")
        df_expl = pd.DataFrame(sc["explication"])[["facteur", "valeur", "reference", "contribution_pts"]]
        df_expl.columns = ["Facteur", "Valeur actuelle", "Référence saine", "Contribution (pts)"]
        tableau(df_expl.set_index("Facteur"))
        top = sc["explication"][0]
        st.info(f"Premier levier : **{top['facteur']}** ({top['valeur']} contre {top['reference']} de référence) "
                f"explique à lui seul {top['contribution_pts']} points de risque.")
    elif e == 3:
        st.markdown("**Simulateur what-if** — que se passe-t-il si on change une habitude ?")
        gain_h = st.slider("Heures de sommeil en plus", 0.0, 3.0, 1.5, 0.5, key="demo_sommeil")
        res_demo = simuler_scenario(sc["features"], {"sommeil_h": _SOMMEIL_DEMO + gain_h})
        w1, w2, w3 = st.columns(3)
        w1.metric("Score avant", f"{res_demo['avant_pct']} / 100")
        w2.metric("Score après", f"{res_demo['apres_pct']} / 100")
        w3.metric("Gain", f"-{res_demo['gain_pts']} pts")
        proba_avant = sc["profil_risque"]["probabilite_prevalence_pct"]  # même valeur qu'à l'étape 1
        proba_apres = corriger_prevalence(res_demo["apres_pct"] / 100) * 100
        st.caption(f"Probabilité estimée : {proba_avant:.1f} % → {proba_apres:.1f} %. "
                   "Calculé par le modèle sur le profil de cette athlète, pas une règle écrite à la main.")
    elif e == 4:
        st.markdown("**Plan d'action chiffré** (leviers classés par gain de risque)")
        for lv in sc["plan_action"]["leviers"]:
            st.markdown(f"- **{lv['action']}** ({lv['de']} → {lv['vers']}) : -{lv['gain_pts']} pts")
        pc = sc["plan_action"]["plan_combine"]
        st.success(f"Plan combiné : score de risque {pc['avant_pct']} → {pc['apres_pct']} / 100 (-{pc['gain_pts']} pts).")
        st.markdown("**Recommandations du coach IA**")
        for reco in sc["recommandations"][:4]:
            st.markdown(f"- **{reco['categorie']}** — {reco['texte']}")
    else:
        st.markdown("**Rapport exportable** pour le staff technique.")
        if st.button("Générer le rapport PDF de la démo"):
            try:
                chemin_demo = generer_pdf_demo(sc, etude=etude_scientifique_en_cache())
                with open(chemin_demo, "rb") as f_pdf:
                    st.download_button("Télécharger le rapport", f_pdf.read(), "ATHENA_Demo.pdf", "application/pdf")
                st.success("Rapport généré.")
            except Exception as err:  # noqa: BLE001
                st.error(f"Échec de la génération du rapport. Détail : {err}")

    n1, n2, n3 = st.columns([1, 1, 4])
    if n1.button("◀ Précédent", disabled=e == 1, key="demo_prec"):
        st.session_state.etape_demo -= 1
        st.rerun()
    if n2.button("Suivant ▶", disabled=e == len(etapes), key="demo_suiv"):
        st.session_state.etape_demo += 1
        st.rerun()


# ONGLET 11 : TEST CSV (épreuve « test du modèle sur données inédites », sans ligne de commande)
with onglet11:
    import io
    from models import testeur_csv as testeur

    st.subheader("Tester le modèle sur un fichier CSV")
    st.caption("Même moteur que `python -m models.testeur_csv` : association automatique des colonnes "
               "(français ou anglais, séparateur , ou ;), nettoyage comme en production, verdict de "
               "compatibilité, puis ROC-AUC avec IC95 si le fichier contient une colonne de blessure.")

    fichier_test = st.file_uploader("Fichier CSV à tester", type=["csv", "txt"], key="csv_test")
    colonnes_test = []
    if fichier_test is not None:
        for enc in ("utf-8-sig", "latin-1"):
            try:
                colonnes_test = list(pd.read_csv(io.BytesIO(fichier_test.getvalue()), sep=None,
                                                 engine="python", encoding=enc, nrows=5).columns)
                break
            except Exception:  # noqa: BLE001 - encodage suivant
                continue
    AUTO = "Détection automatique"
    cible_choisie = st.selectbox("Colonne de blessure (étiquette 0/1, oui/non…)", [AUTO] + colonnes_test,
                                 help="Laisser en détection automatique sauf si la colonne n'est pas reconnue.")
    b1, b2, _ = st.columns([1, 1.4, 2])
    lancer_test = b1.button("Lancer le test", disabled=fichier_test is None)
    lancer_repetition = b2.button("Répétition : jeu inédit simulé")

    if lancer_test or lancer_repetition:
        dossier_tmp = Path(tempfile.mkdtemp(prefix="athena_csv_"))
        if lancer_repetition:
            chemin_test = testeur.jeu_de_repetition(dossier_tmp)
            nom_test, cible_test = "Jeu de répétition simulé (3 000 profils, colonnes en anglais)", None
        else:
            chemin_test = dossier_tmp / Path(fichier_test.name).name
            chemin_test.write_bytes(fichier_test.getvalue())
            nom_test, cible_test = fichier_test.name, (None if cible_choisie == AUTO else cible_choisie)
        with st.spinner("Analyse du fichier en cours…"):
            try:
                res_test = testeur.analyser(chemin_test, cible=cible_test)
                predictions = res_test.pop("_predictions")
                res_test.pop("_y", None)
                st.session_state["resultat_test_csv"] = {
                    "nom": nom_test, "res": res_test, "rapport": testeur.rapport_md(res_test),
                    # Les scores d'ATHÉNA d'abord, puis les colonnes du fichier.
                    "apercu": predictions[[c for c in predictions.columns if str(c).startswith("athena_")]
                                          + [c for c in predictions.columns if not str(c).startswith("athena_")]].head(50),
                    "predictions": predictions.to_csv(index=False).encode("utf-8"),
                }
            except (ValueError, FileNotFoundError) as err:
                st.session_state["resultat_test_csv"] = {"nom": nom_test, "erreur": str(err)}
            except Exception as err:  # noqa: BLE001 - devant le jury : un message lisible, jamais une trace Python
                st.session_state["resultat_test_csv"] = {
                    "nom": nom_test, "erreur": f"Analyse impossible ({type(err).__name__} : {str(err)[:300]}). "
                                              "Vérifier le format du fichier, ou lancer `python -m models.testeur_csv "
                                              "--test fichier.csv` pour le détail."}
            finally:
                shutil.rmtree(dossier_tmp, ignore_errors=True)

    r_test = st.session_state.get("resultat_test_csv")
    if r_test:
        st.markdown(f"#### Résultat : {r_test['nom']}")
        if "erreur" in r_test:
            st.error(r_test["erreur"])
        else:
            res_t = r_test["res"]
            verdict = res_t["verdict"]
            message = (f"Fichier **{verdict['statut'].upper()}** · {res_t['lignes']:,} lignes · "
                       f"{res_t['variables_mesurees']}/10 variables du modèle trouvées").replace(",", " ")
            {"compatible": st.success, "partiel": st.warning, "incompatible": st.error}[verdict["statut"]](message)
            for raison in verdict["raisons"]:
                st.markdown(f"- {raison}")
            if verdict["statut"] == "incompatible":
                st.info("Les scores sont calculés pour la transparence, mais ils ne doivent pas être interprétés : "
                        "sans les variables clés, le modèle n'a presque rien sur quoi s'appuyer. Un fichier de "
                        "résultats sportifs (médailles, palmarès) ne décrit ni la charge, ni le sommeil, ni les blessures.")

            if "production" in res_t:
                p = res_t["production"]
                st.markdown(f"**Risque de blessure** (étiquette : « {res_t['colonne_etiquette']} »)")
                if "auc" in p:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("ROC-AUC", f"{p['auc']:.3f}", help=f"IC95 bootstrap [{p['ic95'][0]:.3f} ; {p['ic95'][1]:.3f}]")
                    m2.metric("IC95", f"[{p['ic95'][0]:.3f} ; {p['ic95'][1]:.3f}]")
                    m3.metric("Rappel, 20 % les plus à risque", f"{p['top20pct']['rappel']:.0%}",
                              help=f"{p['top20pct']['lift']}× mieux que le hasard")
                    m4.metric("Brier (proba corrigée)", f"{p['brier_proba_corrigee']:.4f}",
                              help=f"Référence constante : {p['brier_constante']:.4f}")
                    st.caption(f"n = {p['n']} · prévalence {p['prevalence']:.1%} · AUC = 0,5 : hasard, 1 : parfait.")
                if "avertissement" in p:
                    st.warning(p["avertissement"])
            else:
                st.info("Pas de colonne de blessure détectée : prédictions seules, pas de métrique. "
                        "Choisir la colonne dans la liste ci-dessus si elle existe sous un autre nom.")
            if "fatigue" in res_t and "mae" in res_t["fatigue"]:
                f_ = res_t["fatigue"]
                st.markdown(f"**Fatigue** (colonne « {res_t['colonne_fatigue']} ») : MAE {f_['mae']:.3f} · "
                            f"RMSE {f_['rmse']:.3f} · R² {f_['r2']:.3f} (n = {f_['n']})")
            for avert in res_t["avertissements"]:
                st.caption(f"Remarque : {avert}")

            with st.expander("Correspondance des colonnes et qualité des données"):
                lignes_corr = []
                for feat in testeur.FEATURE_NAMES:
                    a = res_t["correspondance"][feat]
                    q = res_t["qualite"][feat]
                    lignes_corr.append({
                        "Variable du modèle": testeur.LABELS_LISIBLES[feat], "Colonne du fichier": a["colonne"] or "absente",
                        "Méthode": a["methode"], "Manquantes": q["manquantes"],
                        "Hors domaine (bornées)": q["sous_domaine"] + q["sur_domaine"],
                    })
                _pleine_largeur(st.dataframe, pd.DataFrame(lignes_corr), hide_index=True)
            st.markdown("**Aperçu des prédictions** (50 premières lignes)")
            tableau(r_test["apercu"])
            t1, t2, _ = st.columns([1, 1, 2])
            t1.download_button("Télécharger les prédictions (CSV)", r_test["predictions"],
                               "athena_predictions.csv", "text/csv")
            t2.download_button("Télécharger le rapport (Markdown)", r_test["rapport"].encode("utf-8"),
                               "athena_rapport_test.md", "text/markdown")


# PIED DE PAGE
pied_de_page()

# Tout est affiché : l'écran de démarrage s'efface en fondu.
if demarrage:
    demarrage.terminer()
