# ATHÉNA v2 — Plateforme IA sportive prédictive
# Développée par ARiA Tech pour la Pan-African Robotics Competition (PARC 2026)
# Ligue Intelligence Artificielle
# Lancement : streamlit run dashboard/app.py   (depuis la racine du projet)

import sys
import tempfile
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Rend le package "athena" importable quel que soit le répertoire de lancement
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.simulators import (
    generer_signal_ecg, simuler_constantes_vitales, generer_historique_charge,
    generer_micronutriments_simules, generer_snapshot_equipe, EQUIPE_NATIONALE_DEMO,
)
from analytics.metrics import (
    calculer_acwr, calculer_bmr_tdee, moteur_de_decision, calculer_performance_score,
)
from analytics.ai_coach import generer_recommandations_coach
from models.injury_risk import construire_vecteur_features, predire_profil_risque, predire_tendance_risque
from models.experiment_comparaison import charger_ou_executer_experience
from models.explicabilite import expliquer_prediction, recommander_leviers, simuler_scenario
from reports.pdf_report import generer_rapport_pdf
from data.donnees_reelles import (
    COLONNES as COLONNES_JOURNAL, DESCRIPTION_COLONNES, charger_journal, ajouter_entree,
    importer_csv, gabarit_csv, risque_actuel, evaluer_sur_donnees_reelles, DOSSIER_REEL,
)
from dashboard.theme import (
    PALETTE, MODELE_PLOTLY, injecter_theme, marque_barre_laterale, en_tete,
    carte_kpi, carte_alerte, pied_de_page,
)


# CONFIGURATION GÉNÉRALE
st.set_page_config(
    page_title="ATHÉNA | ARiA Tech - PARC 2026",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


injecter_theme()


# ÉTAT DE SESSION
for cle, defaut in [
    ("historique_vitals", deque(maxlen=200)),
    ("dernier_etat_vital", None),
    ("repas_journee", pd.DataFrame(columns=["Heure", "Repas", "Kcal", "Protéines (g)", "Glucides (g)", "Lipides (g)"])),
    ("simulation_active", False),
    ("historique_charge", None),
    ("micronutriments", None),
]:
    if cle not in st.session_state:
        st.session_state[cle] = defaut

if st.session_state.historique_charge is None:
    st.session_state.historique_charge = generer_historique_charge()
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

    sexe = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
    age = st.number_input("Âge", min_value=10, max_value=60, value=int(ref["age"]), key=f"age_{cle}")
    poids = st.number_input("Poids (kg)", min_value=30.0, max_value=150.0, value=float(ref["poids"]), key=f"poids_{cle}")
    taille = st.number_input("Taille (cm)", min_value=120.0, max_value=220.0, value=float(ref["taille"]), key=f"taille_{cle}")

    st.subheader("Données pour l'IA prédictive")
    fc_repos = st.number_input("FC de repos (bpm)", min_value=35, max_value=100, value=60, key=f"fc_{cle}")
    sommeil_h = st.slider("Sommeil moyen (h/nuit)", 3.0, 10.0, float(ref["sommeil_h"]), 0.1, key=f"sommeil_{cle}")
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
df_charge = st.session_state.historique_charge
acwr_info = calculer_acwr(df_charge)
bmr_tdee = calculer_bmr_tdee(sexe, poids, taille, age, facteur_activite)
kcal_consommees = st.session_state.repas_journee["Kcal"].sum() if not st.session_state.repas_journee.empty else 0
bilan_calorique = {"tdee": bmr_tdee["tdee"], "kcal_consommees": kcal_consommees,
                   "renseigne": not st.session_state.repas_journee.empty}
df_micro = st.session_state.micronutriments

vitals_courants = st.session_state.dernier_etat_vital or simuler_constantes_vitales(profil_athlete)
imc = poids / ((taille / 100) ** 2)
deficit_calorique_pct = (
    (bilan_calorique["kcal_consommees"] - bilan_calorique["tdee"]) / bilan_calorique["tdee"] * 100
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


# EN-TÊTE (après les calculs : il affiche le statut de l'athlète)
en_tete(nom_athlete, profil_risque, acwr_info, mode_simulation)


# ONGLETS PRINCIPAUX
onglets = st.tabs([
    "Démo Guidée", "IA Prédictive", "Monitoring", "Charge & ACWR", "Nutrition",
    "Decision Engine", "Analyse Vidéo", "National Team", "Étude Scientifique", "Données Réelles",
])
(onglet10, onglet5, onglet1, onglet2, onglet3, onglet4, onglet6, onglet7, onglet8, onglet9) = onglets

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

    with st.expander("Ajouter une séance du jour"):
        nouvelle_charge = st.slider("Charge de la séance (Session-RPE x durée)", 0, 900, 400)
        if st.button("Enregistrer la séance"):
            nouvelle_ligne = pd.DataFrame({"date": [datetime.now().date()], "charge": [nouvelle_charge]})
            st.session_state.historique_charge = pd.concat([st.session_state.historique_charge, nouvelle_ligne], ignore_index=True)
            st.success("Séance enregistrée. Le calcul de l'ACWR a été mis à jour.")
            st.rerun()


# ONGLET 3 : NUTRITION & MÉTABOLISME
with onglet3:
    st.subheader("Suivi nutritionnel et métabolique")
    col_n1, col_n2, col_n3 = st.columns(3)
    col_n1.metric("BMR (métabolisme de base)", f"{bmr_tdee['bmr']} kcal")
    col_n2.metric("TDEE (besoin journalier)", f"{bmr_tdee['tdee']} kcal")
    col_n3.metric("Kcal consommées", f"{int(kcal_consommees)} kcal",
                  delta=f"{int(kcal_consommees - bmr_tdee['tdee'])} kcal vs objectif")

    st.markdown("#### Enregistrer un repas")
    with st.form("form_repas", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        nom_repas = c1.text_input("Repas", placeholder="Ex : Petit-déjeuner")
        kcal_repas = c2.number_input("Kcal", min_value=0, value=0, step=10)
        proteines = c3.number_input("Protéines (g)", min_value=0, value=0, step=1)
        glucides = c4.number_input("Glucides (g)", min_value=0, value=0, step=1)
        lipides = c5.number_input("Lipides (g)", min_value=0, value=0, step=1)
        valider = st.form_submit_button("Enregistrer le repas")
        if valider and nom_repas:
            nouvelle_entree = pd.DataFrame({
                "Heure": [datetime.now().strftime("%H:%M")], "Repas": [nom_repas], "Kcal": [kcal_repas],
                "Protéines (g)": [proteines], "Glucides (g)": [glucides], "Lipides (g)": [lipides],
            })
            st.session_state.repas_journee = pd.concat([st.session_state.repas_journee, nouvelle_entree], ignore_index=True)
            st.success(f"Repas « {nom_repas} » enregistré.")

    if not st.session_state.repas_journee.empty:
        st.markdown("#### Repas enregistrés aujourd'hui")
        tableau(st.session_state.repas_journee)
        col_macro1, col_macro2 = st.columns(2)
        with col_macro1:
            total_macros = st.session_state.repas_journee[["Protéines (g)", "Glucides (g)", "Lipides (g)"]].sum()
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
        carte_kpi(f"{profil_risque['injury_risk_pct']} %", "Risque de blessure",
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
                                xaxis_title=None, yaxis_title="Injury Risk (%)")
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
            st.caption(f"Gradient Boosting (scikit-learn), profondeur 2, entraîné sur "
                       f"{meta.get('n_train_samples')} profils simulés. Le détail des facteurs propres "
                       f"à cet athlète est présenté plus bas.")

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
                            xaxis_title="Contribution au risque (points de %)")
    tracer(fig_expl)

    st.markdown("#### Plan d'action chiffré")
    if plan_action["leviers"]:
        for lv in plan_action["leviers"]:
            st.markdown(f"- **{lv['action']}** ({lv['de']} {lv['vers']}) : **−{lv['gain_pts']} pts** de risque")
        pc = plan_action["plan_combine"]
        st.success(f"Plan combiné : risque {pc['avant_pct']} % **{pc['apres_pct']} %** (−{pc['gain_pts']} pts).")
    else:
        st.info("Aucun levier à fort impact : le profil est déjà proche de l'optimum sur les facteurs actionnables.")

    with st.expander("Simulateur What-If — testez un scénario"):
        c_w1, c_w2, c_w3 = st.columns(3)
        w_sommeil = c_w1.slider("Sommeil (h)", 3.5, 10.5, float(min(max(sommeil_h, 3.5), 10.5)), 0.1, key="w_sommeil")
        w_acwr = c_w2.slider("ACWR", 0.3, 2.5, float(min(max(acwr_info["acwr"], 0.3), 2.5)), 0.05, key="w_acwr")
        w_int = c_w3.slider("Intensité moyenne 7j", 0.05, 1.0, float(min(max(intensite_effort, 0.05), 1.0)), 0.05, key="w_int")
        res_w = simuler_scenario(features, {"sommeil_h": w_sommeil, "acwr": w_acwr, "intensite_moyenne_7j": w_int})
        m1, m2, m3 = st.columns(3)
        m1.metric("Risque actuel", f"{res_w['avant_pct']} %")
        m2.metric("Risque simulé", f"{res_w['apres_pct']} %", delta=f"{-res_w['gain_pts']:+.1f} pts", delta_color="inverse")
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

    from computer_vision.pose_analysis import MOUVEMENTS_REFERENCE, modele_disponible_localement, telecharger_modele

    if not modele_disponible_localement():
        st.warning(
            "Le modèle de détection posturale n'est pas encore présent en local sur cette machine "
            "(il sera téléchargé une seule fois, ~5 Mo, à la première analyse — connexion requise). "
            "Pour une démo sans dépendre du wifi le jour J : lancer une fois "
            "`python -m computer_vision.pose_analysis --setup` avant l'événement."
        )

    if not modele_disponible_localement():
        if st.button("Télécharger le modèle de détection (~5 Mo)"):
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

    fichier_image = st.file_uploader("Image de l'athlète (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if fichier_image is not None:
        import cv2
        from computer_vision.pose_analysis import analyser_image, dessiner_squelette

        octets_image = np.frombuffer(fichier_image.read(), np.uint8)
        image_bgr = cv2.imdecode(octets_image, cv2.IMREAD_COLOR)

        resultat = None
        if image_bgr is None:
            st.error("Image illisible ou fichier corrompu. Réessayez avec un JPG/PNG valide.")
        else:
            try:
                with st.spinner("Analyse posturale en cours..."):
                    resultat = analyser_image(image_bgr, mouvement=mouvement_analyse)
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
        else:
            col_v1, col_v2 = st.columns([1.1, 1])
            with col_v1:
                image_annotee = dessiner_squelette(image_bgr, resultat["landmarks_bruts"])
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
    fichier_video = st.file_uploader("Vidéo de l'athlète (MP4/MOV/AVI)", type=["mp4", "mov", "avi"], key="upload_video_seq")
    if fichier_video is not None:
        from computer_vision.analyse_sequence import analyser_video
        with tempfile.NamedTemporaryFile(suffix=Path(fichier_video.name).suffix, delete=False) as tmp_video:
            tmp_video.write(fichier_video.read())
            chemin_tmp_video = tmp_video.name
        resultat_seq = None
        try:
            barre = st.progress(0.0, text="Analyse de la séquence en cours…")
            resultat_seq = analyser_video(chemin_tmp_video, mouvement=mouvement_analyse,
                                          rappel_progression=lambda p: barre.progress(min(1.0, p)))
            barre.empty()
        except ValueError as e:
            st.error(f"Vidéo non exploitable : {e}")
        except Exception as e:  # noqa: BLE001 - modèle de pose absent, etc.
            st.error("L'analyse n'a pas pu s'exécuter (modèle de pose absent ? lancer "
                     f"`python -m computer_vision.pose_analysis --setup` avec internet). Détail : {e}")
        finally:
            Path(chemin_tmp_video).unlink(missing_ok=True)

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
                           f"({resultat_seq['fps']} images/s). Aucune vidéo n'est conservée sur le disque après l'analyse.")

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
            "Injury Risk (%)": profil["injury_risk_pct"], "Fatigue (/10)": profil["fatigue_score"],
            "Overtraining (%)": profil["overtraining_pct"], "Disponibilité": disponibilite,
        })

    df_classement = pd.DataFrame(lignes_scored).sort_values("Injury Risk (%)")

    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("Effectif suivi", len(df_classement))
    col_e2.metric("Athlètes à surveiller", (df_classement["Disponibilité"] == "À surveiller").sum())
    col_e3.metric("Injury Risk moyen", f"{df_classement['Injury Risk (%)'].mean():.1f} %")

    st.markdown("#### Classement par risque (du plus sûr au plus exposé)")
    _pleine_largeur(
        st.dataframe,
        df_classement.style.background_gradient(subset=["Injury Risk (%)"], cmap="RdYlGn_r", vmin=0, vmax=100)
        .format({"ACWR": "{:.2f}", "Injury Risk (%)": "{:.1f}", "Fatigue (/10)": "{:.1f}", "Overtraining (%)": "{:.1f}"}),
        hide_index=True,
    )

    fig_equipe = px.bar(df_classement, x="Athlète", y=["Injury Risk (%)", "Overtraining (%)"],
                         barmode="group", template=MODELE_PLOTLY, title="Risque de blessure vs surentraînement par athlète",
                         labels={"value": "%", "variable": ""})
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
    journal = charger_journal()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Athlètes suivis", journal["athlete_id"].nunique() if len(journal) else 0)
    col_b.metric("Jours saisis", len(journal))
    col_c.metric("Blessures déclarées", int(journal["blessure"].fillna(0).sum()) if len(journal) else 0)

    st.markdown("#### 1. Saisir une journée")
    with st.form("form_journal", clear_on_submit=False):
        f1, f2, f3 = st.columns(3)
        id_saisie = f1.text_input("Identifiant (ex. A01)", value="A01")
        date_saisie = f2.date_input("Date", value=datetime.now().date())
        rpe_saisie = f3.slider("Effort perçu de la séance (0 = repos)", 0, 10, 5)
        g1, g2, g3 = st.columns(3)
        duree_saisie = g1.number_input("Durée séance (min)", 0, 300, 60)
        sommeil_saisie = g2.number_input("Sommeil (h)", 2.0, 14.0, 7.5, step=0.25)
        fc_saisie = g3.number_input("FC au réveil (bpm)", 30, 120, 60)
        h1, h2, h3 = st.columns(3)
        poids_saisie = h1.number_input("Poids (kg, optionnel)", 30.0, 200.0, 70.0, step=0.1)
        douleur_saisie = h2.slider("Douleur / courbatures (0-10)", 0, 10, 0)
        blessure_saisie = h3.checkbox("Blessure survenue ce jour")
        if st.form_submit_button("Enregistrer la journée"):
            ajouter_entree({
                "athlete_id": id_saisie.strip(), "date": date_saisie, "rpe": rpe_saisie,
                "duree_min": duree_saisie, "sommeil_h": sommeil_saisie, "fc_repos": fc_saisie,
                "poids_kg": poids_saisie, "douleur": douleur_saisie, "blessure": int(blessure_saisie),
            })
            st.success("Journée enregistrée.")
            st.rerun()

    st.markdown("#### 2. Importer / exporter")
    i1, i2 = st.columns(2)
    with i1:
        fichier = st.file_uploader("Importer un journal CSV", type="csv")
        if fichier is not None and st.button("Fusionner ce fichier"):
            _, rap_import = importer_csv(fichier)
            st.success(f"{rap_import['lignes_valides']} lignes valides importées "
                       f"({rap_import['valeurs_hors_plage']} valeurs hors plage ignorées, "
                       f"{rap_import['doublons_supprimes']} doublons).")
            st.rerun()
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
        choix_athlete = st.selectbox("Athlète", sorted(journal["athlete_id"].unique()))
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
            st.warning(f"Historique insuffisant : {res['jours']} jour(s) saisi(s), {res['jours_requis']} requis "
                       "pour calculer des features fiables.")
        else:
            pr = res["profil"]
            r1, r2, r3 = st.columns(3)
            r1.metric("Risque de blessure", f"{pr['injury_risk_pct']} %")
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
        d1, d2, d3 = st.columns(3)
        niveau_demo = sc["profil_risque"]["injury_risk_categorie"]["niveau"]
        d1.metric("Risque de blessure", f"{sc['profil_risque']['injury_risk_pct']} %", niveau_demo,
                  delta_color="normal" if niveau_demo == "Faible" else "inverse")
        d2.metric("ACWR", float(sc["acwr_info"]["acwr"]), sc["acwr_info"]["zone"], delta_color="off")
        d3.metric("Fatigue", f"{sc['profil_risque']['fatigue_score']} / 10")
        for niveau, titre, action in sc["alertes"][:3]:
            (st.error if niveau == "Critique" else st.warning)(f"**{titre}** — {action}")
    elif e == 2:
        st.markdown("**Qu'est-ce qui pèse sur son risque ?** (points de % vs une référence saine)")
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
        w1.metric("Avant", f"{res_demo['avant_pct']} %")
        w2.metric("Après", f"{res_demo['apres_pct']} %")
        w3.metric("Gain", f"-{res_demo['gain_pts']} pts")
        st.caption("Calculé par le modèle sur le profil de cette athlète, pas une règle écrite à la main.")
    elif e == 4:
        st.markdown("**Plan d'action chiffré** (leviers classés par gain de risque)")
        for lv in sc["plan_action"]["leviers"]:
            st.markdown(f"- **{lv['action']}** ({lv['de']} {lv['vers']}) : -{lv['gain_pts']} pts")
        pc = sc["plan_action"]["plan_combine"]
        st.success(f"Plan combiné : risque {pc['avant_pct']} % {pc['apres_pct']} % (-{pc['gain_pts']} pts).")
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


# PIED DE PAGE
pied_de_page()
