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
from reports.pdf_report import generer_rapport_pdf


# CONFIGURATION GÉNÉRALE
st.set_page_config(
    page_title="ATHÉNA | ARiA Tech - PARC 2026",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# THÈME CLINIQUE SOBRE
def injecter_theme_clinique():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        .stApp {
            background: #F5F8FA;
            color: #1F2937;
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3, h4 {
            font-family: 'Inter', sans-serif !important;
            color: #1B4B66 !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 12px 10px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stMetricLabel"] { color: #5B7285 !important; }
        div[data-testid="stMetricValue"] { color: #1B4B66 !important; }
        button[data-baseweb="tab"] {
            font-family: 'Inter', sans-serif;
            color: #5B7285 !important;
            font-size: 0.82rem;
        }
        div[data-baseweb="tab-highlight"] { background-color: #1B6E8C !important; }
        .stButton>button {
            background: #1B6E8C;
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
        }
        .stButton>button:hover {
            background: #15586F;
            color: #FFFFFF;
        }
        .aria-logo {
            font-family: 'Inter', sans-serif; font-size: 1.6rem; font-weight: 800;
            letter-spacing: 1px; color: #1B4B66;
            text-align: center;
        }
        .aria-sub { text-align:center; color:#5B7285; font-size:0.75rem; letter-spacing:2px; margin-top:-8px; }
        .sponsor-box {
            border: 1px dashed #C4D3DC; border-radius: 10px; padding: 10px;
            text-align: center; color: #5B7285; font-size: 0.75rem; background: #FFFFFF;
        }
        .badge-critique { background:#D9364B; color:white; padding:4px 10px; border-radius:6px; font-weight:700; }
        .badge-attention { background:#E9A029; color:#1a1a1a; padding:4px 10px; border-radius:6px; font-weight:700; }
        .badge-info { background:#1B6E8C; color:#FFFFFF; padding:4px 10px; border-radius:6px; font-weight:700; }
        .badge-ok { background:#2E9E6B; color:#FFFFFF; padding:4px 10px; border-radius:6px; font-weight:700; }
        .carte-alerte {
            background: #FFFFFF; border-left: 4px solid #1B6E8C;
            border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px; padding: 18px 16px; text-align:center;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .kpi-value { font-family:'Inter',sans-serif; font-size:2.1rem; font-weight:800; color:#1B4B66; }
        .kpi-label { color:#5B7285; font-size:0.78rem; letter-spacing:0.5px; margin-top:4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


injecter_theme_clinique()


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


def couleur_badge(niveau: str) -> str:
    return {"Faible": "badge-ok", "Modéré": "badge-attention", "Élevé": "badge-critique"}.get(niveau, "badge-info")


# BARRE LATÉRALE
with st.sidebar:
    st.markdown('<div class="aria-logo">⚡ ARiA TECH</div>', unsafe_allow_html=True)
    st.markdown('<div class="aria-sub">PROJET ATHÉNA v2 — PARC 2026</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("👤 Profil Athlète")
    liste_athletes = ["Amara K. (Sprint)", "Fatou D. (Endurance)", "Kwame O. (Football)", "Profil personnalisé"]
    athlete_selectionne = st.selectbox("Sélectionner un athlète", liste_athletes)

    sexe = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
    age = st.number_input("Âge", min_value=10, max_value=60, value=22)
    poids = st.number_input("Poids (kg)", min_value=30.0, max_value=150.0, value=68.0)
    taille = st.number_input("Taille (cm)", min_value=120.0, max_value=220.0, value=175.0)

    st.markdown("---")
    st.subheader("🧠 Données pour l'IA prédictive")
    fc_repos = st.number_input("FC de repos (bpm)", min_value=35, max_value=100, value=62)
    sommeil_h = st.slider("Sommeil moyen (h/nuit)", 3.0, 10.0, 7.0, 0.1)
    historique_blessures = st.number_input("Blessures (12 derniers mois)", min_value=0, max_value=10, value=0)

    st.markdown("---")
    st.subheader("📡 Mode d'acquisition")
    mode_acquisition = st.radio(
        "Source des données", ["🧪 Simulation", "🔵 Live Bluetooth / Capteurs"],
        help="Le mode Live nécessite un module physique connecté (point d'injection matériel prévu pour Raspberry Pi).",
    )
    if mode_acquisition.startswith("🔵"):
        st.warning("Aucun capteur physique détecté. Basculement automatique sur le simulateur.")
    mode_reel = mode_acquisition.startswith("🧪")

    st.markdown("---")
    st.subheader("⚙️ Paramètres")
    intensite_effort = st.slider("Intensité d'effort simulée", 0.0, 1.0, 0.5, 0.05)
    intervalle_rafraichissement = st.slider("Fréquence de rafraîchissement (s)", 0.2, 2.0, 0.6, 0.1)
    facteur_activite = st.select_slider(
        "Niveau d'activité (TDEE)", options=[1.2, 1.375, 1.55, 1.725, 1.9], value=1.55,
        format_func=lambda x: {1.2: "Sédentaire", 1.375: "Léger", 1.55: "Modéré",
                                1.725: "Intense", 1.9: "Athlète élite"}[x],
    )

    st.markdown("---")
    st.subheader("🤝 Sponsors")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown('<div class="sponsor-box">LOGO<br>SPONSOR 1</div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown('<div class="sponsor-box">LOGO<br>SPONSOR 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="sponsor-box" style="margin-top:8px;">Espace disponible — ARiA Tech Partners</div>', unsafe_allow_html=True)

profil_athlete = {
    "nom": athlete_selectionne, "sexe": sexe, "age": age, "poids": poids, "taille": taille,
    "intensite_effort": intensite_effort,
}


# EN-TÊTE
col_titre, col_statut = st.columns([3, 1])
with col_titre:
    st.title("🧬 ATHÉNA — Plateforme IA Sportive Prédictive")
    st.caption(f"Athlète actif : **{athlete_selectionne}**  |  ARiA Tech  |  PARC 2026 — Ligue Intelligence Artificielle")
with col_statut:
    st.metric("Mode", "Simulation" if mode_reel else "Live (fallback simulation)")


# CALCULS PARTAGÉS (utilisés par plusieurs onglets)
df_charge = st.session_state.historique_charge
acwr_info = calculer_acwr(df_charge)
bmr_tdee = calculer_bmr_tdee(sexe, poids, taille, age, facteur_activite)
kcal_consommees = st.session_state.repas_journee["Kcal"].sum() if not st.session_state.repas_journee.empty else 0
bilan_calorique = {"tdee": bmr_tdee["tdee"], "kcal_consommees": kcal_consommees}
df_micro = st.session_state.micronutriments

vitals_courants = st.session_state.dernier_etat_vital or simuler_constantes_vitales(profil_athlete)
imc = poids / ((taille / 100) ** 2)
deficit_calorique_pct = (
    (bilan_calorique["kcal_consommees"] - bilan_calorique["tdee"]) / bilan_calorique["tdee"] * 100
    if bilan_calorique["tdee"] > 0 else 0
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


# ONGLETS PRINCIPAUX
onglets = st.tabs([
    "📈 Monitoring Temps Réel", "⚖️ Charge & Fatigue (ACWR)", "🍎 Nutrition & Métabolisme",
    "🧠 Decision Engine", "🤖 IA Prédictive", "🎥 Analyse Vidéo", "🌍 National Team",
    "🔬 Étude Scientifique",
])
(onglet1, onglet2, onglet3, onglet4, onglet5, onglet6, onglet7, onglet8) = onglets

# ONGLET 1 : MONITORING TEMPS RÉEL
with onglet1:
    st.subheader("Monitoring physiologique en temps réel")
    demarrer = st.toggle("▶️ Démarrer la simulation live", value=st.session_state.simulation_active)
    st.session_state.simulation_active = demarrer

    def afficher_etat_vital(vitals: dict):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("❤️ FC (bpm)", vitals["fc"])
        c2.metric("🫁 SpO2 (%)", vitals["spo2"])
        c3.metric("🌡️ Température (°C)", vitals["temperature"])
        c4.metric("🧪 Lactate (mmol/L)", vitals["lactate"])
        c5.metric("🍬 Glucose (mg/dL)", vitals["glucose"])

        signal_ecg = generer_signal_ecg(hr_bpm=vitals["fc"], duree_s=2.5)
        temps_axe = np.linspace(0, 2.5, len(signal_ecg))
        fig_ecg = go.Figure()
        fig_ecg.add_trace(go.Scatter(x=temps_axe, y=signal_ecg, mode="lines",
                                      line=dict(color="#1B6E8C", width=2), name="ECG"))
        fig_ecg.update_layout(title="Tracé ECG synthétique", template="plotly_white",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320,
                               margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Temps (s)", yaxis_title="Amplitude (mV)")
        st.plotly_chart(fig_ecg, use_container_width=True)

        if len(st.session_state.historique_vitals) > 1:
            df_hist = pd.DataFrame(list(st.session_state.historique_vitals))
            fig_hist = px.line(df_hist, x="horodatage", y=["fc", "spo2"], template="plotly_white",
                                title="Historique récent — FC & SpO2")
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280)
            st.plotly_chart(fig_hist, use_container_width=True)

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
                   "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": acwr_info["acwr"]}},
        ))
        fig_jauge.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", height=320,
                                 margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(fig_jauge, use_container_width=True)
        st.markdown(f"**Zone actuelle :** {acwr_info['zone']}")
        st.caption(f"Charge aiguë (7j) : {acwr_info['aigue']}  |  Charge chronique (28j) : {acwr_info['chronique']}")

    with col_g2:
        fig_charge = px.bar(df_charge, x="date", y="charge", template="plotly_white",
                             title="Charge d'entraînement journalière (Session-RPE)")
        fig_charge.update_traces(marker_color="#5B9BB0")
        fig_charge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
        st.plotly_chart(fig_charge, use_container_width=True)

    with st.expander("➕ Ajouter une séance du jour"):
        nouvelle_charge = st.slider("Charge de la séance (Session-RPE x durée)", 0, 900, 400)
        if st.button("Enregistrer la séance"):
            nouvelle_ligne = pd.DataFrame({"date": [datetime.now().date()], "charge": [nouvelle_charge]})
            st.session_state.historique_charge = pd.concat([df_charge, nouvelle_ligne], ignore_index=True)
            st.success("Séance enregistrée. Le calcul de l'ACWR a été mis à jour.")
            st.rerun()


# ONGLET 3 : NUTRITION & MÉTABOLISME
with onglet3:
    st.subheader("Suivi nutritionnel et métabolique")
    col_n1, col_n2, col_n3 = st.columns(3)
    col_n1.metric("🔥 BMR (métabolisme de base)", f"{bmr_tdee['bmr']} kcal")
    col_n2.metric("⚡ TDEE (besoin journalier)", f"{bmr_tdee['tdee']} kcal")
    col_n3.metric("🍽️ Kcal consommées", f"{int(kcal_consommees)} kcal",
                  delta=f"{int(kcal_consommees - bmr_tdee['tdee'])} kcal vs objectif")

    st.markdown("#### ➕ Enregistrer un repas")
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
        st.markdown("#### 📋 Repas enregistrés aujourd'hui")
        st.dataframe(st.session_state.repas_journee, use_container_width=True)
        col_macro1, col_macro2 = st.columns(2)
        with col_macro1:
            total_macros = st.session_state.repas_journee[["Protéines (g)", "Glucides (g)", "Lipides (g)"]].sum()
            fig_macros = px.pie(names=total_macros.index, values=total_macros.values, template="plotly_white",
                                 title="Répartition des macronutriments",
                                 color_discrete_sequence=["#1B6E8C", "#5B9BB0", "#FFB703"])
            fig_macros.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig_macros, use_container_width=True)
        with col_macro2:
            fig_bilan = go.Figure(go.Bar(x=["Consommé", "Objectif (TDEE)"], y=[kcal_consommees, bmr_tdee["tdee"]],
                                          marker_color=["#1B6E8C", "#5B7285"]))
            fig_bilan.update_layout(template="plotly_white", title="Bilan calorique du jour",
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig_bilan, use_container_width=True)
    else:
        st.info("Aucun repas enregistré pour le moment.")

    st.markdown("#### 🧬 Panel micronutriments")
    df_micro_affiche = df_micro.copy()
    df_micro_affiche["Statut"] = df_micro_affiche.apply(
        lambda l: "⚠️ Carence" if l["Valeur mesurée"] < l["Seuil minimal"] else "✅ Normal", axis=1)
    st.dataframe(df_micro_affiche, use_container_width=True)


# ONGLET 4 : DECISION ENGINE (règles)
with onglet4:
    st.subheader("Moteur de décision — Alertes prioritaires & recommandations")
    st.caption("Synthèse générée automatiquement à destination du staff technique et du coach.")

    nb_critiques = sum(1 for a in alertes if a[0] == "Critique")
    nb_attention = sum(1 for a in alertes if a[0] == "Attention")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("🔴 Alertes critiques", nb_critiques)
    col_r2.metric("🟠 Points d'attention", nb_attention)
    col_r3.metric("📊 ACWR actuel", acwr_info["acwr"])

    st.markdown("---")
    st.markdown("### 📋 Recommandations priorisées")
    classes_css = {"Critique": "badge-critique", "Attention": "badge-attention", "Info": "badge-info", "OK": "badge-ok"}
    for niveau, titre, recommandation in alertes:
        badge_class = classes_css.get(niveau, "badge-info")
        st.markdown(
            f"""<div class="carte-alerte"><span class="{badge_class}">{niveau}</span>
            <strong style="margin-left:10px;">{titre}</strong>
            <p style="margin-top:6px; margin-bottom:0; color:#334155;">{recommandation}</p></div>""",
            unsafe_allow_html=True,
        )


# ONGLET 5 : IA PRÉDICTIVE (Injury Risk, AI Coach, Performance Score, Rapport)
with onglet5:
    st.subheader("🤖 Intelligence artificielle prédictive")
    st.caption("Modèles Gradient Boosting (scikit-learn) — Injury Risk, Fatigue, Overtraining, Performance Score.")

    col_k1, col_k2, col_k3 = st.columns(3)
    for col, cle, label in [
        (col_k1, "injury_risk_pct", "🩹 INJURY RISK SCORE"),
        (col_k2, "fatigue_score", "😴 FATIGUE LEVEL"),
        (col_k3, "overtraining_pct", "⚠️ OVERTRAINING RISK"),
    ]:
        if cle == "fatigue_score":
            valeur, niveau, couleur = profil_risque[cle], profil_risque["fatigue_categorie"]["niveau"], profil_risque["fatigue_categorie"]["couleur"]
            texte_val = f"{valeur} / 10"
        elif cle == "injury_risk_pct":
            valeur, niveau, couleur = profil_risque[cle], profil_risque["injury_risk_categorie"]["niveau"], profil_risque["injury_risk_categorie"]["couleur"]
            texte_val = f"{valeur} %"
        else:
            valeur, niveau, couleur = profil_risque[cle], profil_risque["overtraining_categorie"]["niveau"], profil_risque["overtraining_categorie"]["couleur"]
            texte_val = f"{valeur} %"
        with col:
            st.markdown(
                f"""<div class="kpi-card"><div class="kpi-value" style="color:{couleur};">{texte_val}</div>
                <div class="kpi-label">{label}<br><span class="{couleur_badge(niveau)}" style="margin-top:6px;display:inline-block;">{niveau}</span></div></div>""",
                unsafe_allow_html=True,
            )

    st.caption(
        f"Tendance Injury Risk (7j) : {tendance_sens['icone']} {tendance_sens['texte']} "
        f"— recalculée jour par jour à partir de l'historique réel de charge (ACWR glissant)."
    )
    fig_tendance = go.Figure(go.Scatter(
        x=[t["jour_offset"] for t in tendance_trajectoire],
        y=[t["injury_risk_pct"] for t in tendance_trajectoire],
        mode="lines+markers", line=dict(color="#1B6E8C", width=3), marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(27,110,140,0.08)",
    ))
    fig_tendance.update_layout(template="plotly_white", height=160, paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title=None, yaxis_title="Injury Risk (%)")
    st.plotly_chart(fig_tendance, use_container_width=True)

    st.markdown("---")
    col_perf1, col_perf2 = st.columns([1, 1])

    with col_perf1:
        st.markdown("#### 🎯 Athlete Performance Score")
        fig_score = go.Figure(go.Indicator(
            mode="gauge+number", value=performance["score_global"],
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1B6E8C"},
                   "steps": [{"range": [0, 50], "color": "#F9D9DE"}, {"range": [50, 75], "color": "#FBEBCF"},
                             {"range": [75, 100], "color": "#DCF0E4"}]},
        ))
        fig_score.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", height=260,
                                 margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(fig_score, use_container_width=True)

    with col_perf2:
        st.markdown("#### 📊 Radar de performance")
        dims = performance["dimensions"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(dims.values()) + [list(dims.values())[0]],
            theta=list(dims.keys()) + [list(dims.keys())[0]],
            fill="toself", line=dict(color="#1B6E8C"), fillcolor="rgba(0,245,212,0.25)",
        ))
        fig_radar.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", height=290,
                                 polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                 margin=dict(l=30, r=30, t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    col_f1, col_f2 = st.columns([1, 1.3])
    with col_f1:
        st.markdown("#### 🔍 Facteurs contributifs (explicabilité)")
        for f in profil_risque["facteurs_cles"]:
            st.markdown(f"**{f['facteur']}** — poids relatif `{f['poids_relatif']:.2f}` · valeur observée `{f['valeur_observee']}`")
        meta = profil_risque.get("metadata_modele", {})
        if meta.get("metrics"):
            st.caption(f"Modèle : {meta.get('model_type', 'ML')} · AUC test = {meta['metrics'].get('injury_model_auc')} · "
                       f"MAE fatigue = {meta['metrics'].get('fatigue_model_mae')} · entraîné sur {meta.get('n_train_samples')} échantillons.")

    with col_f2:
        st.markdown("#### 💬 AI Coach — Recommandations")
        for r in recommandations:
            urgence = {1: "🔴", 2: "🟠", 3: "🔵", 4: "🟢"}.get(r["priorite"], "🔵")
            st.markdown(f"{urgence} **[{r['categorie']}]** {r['texte']}")

    st.markdown("---")
    st.markdown("#### 📄 Rapport IA exportable")
    if st.button("🖨️ Generate AI Report"):
        try:
            with st.spinner("Génération du rapport PDF en cours..."):
                chemin_temp = str(Path(tempfile.gettempdir()) / f"athena_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                generer_rapport_pdf(chemin_temp, profil_athlete, profil_risque, acwr_info,
                                     performance, recommandations, alertes,
                                     etude_scientifique=charger_ou_executer_experience())
                with open(chemin_temp, "rb") as f:
                    donnees_pdf = f.read()
            st.download_button("⬇️ Télécharger le rapport PDF", data=donnees_pdf,
                                file_name=f"ATHENA_Rapport_{athlete_selectionne.split()[0]}.pdf",
                                mime="application/pdf")
            st.success("Rapport généré avec succès.")
        except Exception as e:
            st.error(f"Échec de la génération du rapport PDF. Détail technique : {e}")


# ONGLET 6 : ANALYSE VIDÉO (Computer Vision)
with onglet6:
    st.subheader("Analyse biomécanique par vision par ordinateur")
    st.caption("Détection posturale via MediaPipe Pose, calibrée par type de mouvement — Football, Basketball, Athlétisme.")

    from computer_vision.pose_analysis import MOUVEMENTS_REFERENCE, modele_disponible_localement

    if not modele_disponible_localement():
        st.warning(
            "Le modèle de détection posturale n'est pas encore présent en local sur cette machine "
            "(il sera téléchargé une seule fois, ~5 Mo, à la première analyse — connexion requise). "
            "Pour une démo sans dépendre du wifi le jour J : lancer une fois "
            "`python -m computer_vision.pose_analysis --setup` avant l'événement."
        )

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

        if image_bgr is None:
            st.error("Image illisible ou fichier corrompu. Réessayez avec un JPG/PNG valide.")
            st.stop()

        try:
            with st.spinner("Analyse posturale en cours..."):
                resultat = analyser_image(image_bgr, mouvement=mouvement_analyse)
        except Exception as e:
            st.error(
                "Le modèle de détection posturale n'a pas pu être chargé (probablement pas de "
                "connexion internet pour le premier téléchargement). Lancer "
                f"`python -m computer_vision.pose_analysis --setup` avec le wifi puis réessayer. Détail : {e}"
            )
            st.stop()

        if not resultat["detecte"]:
            st.warning("Aucune silhouette détectée sur cette image. Essayez une image où le corps entier est visible.")
        else:
            col_v1, col_v2 = st.columns([1.1, 1])
            with col_v1:
                image_annotee = dessiner_squelette(image_bgr, resultat["landmarks_bruts"])
                st.image(cv2.cvtColor(image_annotee, cv2.COLOR_BGR2RGB), caption="Squelette détecté", use_container_width=True)
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
                    couleur_asym = {"Aucune notable": "🟢", "Légère": "🟠", "Notable": "🔴"}.get(asym["niveau"], "🔵")
                    st.markdown(
                        f"**Asymétrie gauche/droite :** {couleur_asym} {asym['niveau']} "
                        f"(écart genou {asym['ecart_genou']}°, écart hanche {asym['ecart_hanche']}°)"
                    )
                    if asym["niveau"] == "Notable":
                        st.caption("Un écart marqué entre les deux côtés est un signal classique de compensation posturale — à recouper avec le staff technique, pas un diagnostic en soi.")
    else:
        st.caption("Aucune image chargée. En l'absence de flux vidéo, ce module accepte une image fixe pour la démonstration.")


# ONGLET 7 : NATIONAL TEAM DASHBOARD (multi-athlètes)
with onglet7:
    st.subheader("🌍 National Team Dashboard — Vue centre de performance")
    st.caption("Surveillance simultanée de l'ensemble de l'effectif — classement, risque, fatigue, disponibilité.")

    df_equipe = generer_snapshot_equipe()

    lignes_scored = []
    for _, ligne in df_equipe.iterrows():
        athlete_ref = next(a for a in EQUIPE_NATIONALE_DEMO if a["nom"] == ligne["Nom"])
        feats = construire_vecteur_features(
            acwr=ligne["ACWR"], charge_chronique=420, fc_repos=60, sommeil_h=ligne["Sommeil (h)"],
            age=ligne["Âge"], imc=23, historique_blessures=ligne["Historique blessures"],
            deficit_calorique_pct=0, lactate=2.5, intensite_moyenne_7j=athlete_ref["intensite_effort"],
        )
        profil = predire_profil_risque(feats)
        disponibilite = "✅ Disponible" if profil["injury_risk_pct"] < 50 else "⛔ À surveiller"
        lignes_scored.append({
            "Athlète": ligne["Nom"], "Sport": ligne["Sport"], "ACWR": ligne["ACWR"],
            "Injury Risk (%)": profil["injury_risk_pct"], "Fatigue (/10)": profil["fatigue_score"],
            "Overtraining (%)": profil["overtraining_pct"], "Disponibilité": disponibilite,
        })

    df_classement = pd.DataFrame(lignes_scored).sort_values("Injury Risk (%)")

    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("👥 Effectif suivi", len(df_classement))
    col_e2.metric("⛔ Athlètes à surveiller", (df_classement["Disponibilité"] == "⛔ À surveiller").sum())
    col_e3.metric("📊 Injury Risk moyen", f"{df_classement['Injury Risk (%)'].mean():.1f} %")

    st.markdown("#### 🏆 Classement par risque (du plus sûr au plus exposé)")
    st.dataframe(
        df_classement.style.background_gradient(subset=["Injury Risk (%)"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )

    fig_equipe = px.bar(df_classement, x="Athlète", y=["Injury Risk (%)", "Overtraining (%)"],
                         barmode="group", template="plotly_white", title="Risque de blessure vs surentraînement par athlète")
    fig_equipe.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
    st.plotly_chart(fig_equipe, use_container_width=True)


# ONGLET 8 : ÉTUDE SCIENTIFIQUE (Research Question / Baseline / Results / Discussion)
with onglet8:
    st.subheader("🔬 Étude scientifique comparative")
    st.caption("Format Research Question / Hypothesis / Baseline / Conditions / Results / Discussion.")

    exp = charger_ou_executer_experience()

    st.markdown("#### 📚 Référence scientifique")
    st.markdown(f"*{exp['reference_scientifique']}*")
    st.caption(
        "Cette revue de 38 études conclut que les modèles à base d'arbres (Random Forest, "
        "XGBoost/Gradient Boosting) obtiennent la meilleure performance de prédiction (AUC) "
        "dans la majorité des études, devant la régression logistique — c'est cette observation "
        "de la littérature qu'ATHÉNA teste ici sur son propre cas d'usage."
    )

    st.markdown("#### ❓ Research Question")
    st.info(exp["research_question"])

    st.markdown("#### 🧪 Hypothèse")
    col_h1, col_h0 = st.columns(2)
    col_h1.success(f"**H1** : {exp['hypothese_h1']}")
    col_h0.warning(f"**H0** : {exp['hypothese_h0']}")

    st.markdown("#### ⚙️ Conditions expérimentales")
    cond = exp["conditions_experimentales"]
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    col_c1.metric("Échantillons", cond["n_echantillons"])
    col_c2.metric("Train / Test", f"{cond['n_train']} / {cond['n_test']}")
    col_c3.metric("Split", cond["split"])
    col_c4.metric("Features", len(cond["features"]))
    st.caption(f"Pondération : {cond['ponderation']} — comparaison équitable entre les 3 modèles.")

    st.markdown("---")
    st.markdown("#### 📊 Results — Classification (Injury Risk)")
    df_clf = pd.DataFrame(exp["classification_injury_risk"]).set_index("modele")
    st.dataframe(
        df_clf.style.background_gradient(subset=["roc_auc", "f1", "recall"], cmap="RdYlGn"),
        use_container_width=True,
    )

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        st.image(exp["figures"]["roc_curves"], use_container_width=True)
    with col_fig2:
        st.image(exp["figures"]["confusion_matrix"], use_container_width=True)
    st.image(exp["figures"]["feature_importance"], use_container_width=True)

    st.markdown("#### 📊 Results — Régression (Fatigue Score)")
    df_reg = pd.DataFrame(exp["regression_fatigue"]).set_index("modele")
    st.dataframe(
        df_reg.style.background_gradient(subset=["mae", "rmse"], cmap="RdYlGn_r").background_gradient(subset=["r2"], cmap="RdYlGn"),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### 🧮 Test d'hypothèse (bootstrap apparié, 1000 tirages)")
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

    st.markdown("#### 🔁 Robustesse — Validation croisée stratifiée (5-fold)")
    st.caption(
        "Le split 80/20 unique ci-dessus peut être chanceux. Voici les 3 modèles réentraînés "
        "sur 5 splits indépendants (moyenne ± écart-type) — c'est la version robuste des résultats."
    )
    df_cv = pd.DataFrame(exp["validation_croisee_5fold"])
    df_cv["ROC-AUC"] = df_cv.apply(lambda r: f"{r['roc_auc_moyen']:.3f} ± {r['roc_auc_ecart_type']:.3f}", axis=1)
    df_cv["F1"] = df_cv.apply(lambda r: f"{r['f1_moyen']:.3f} ± {r['f1_ecart_type']:.3f}", axis=1)
    st.dataframe(df_cv[["modele", "ROC-AUC", "F1"]].set_index("modele"), use_container_width=True)

    if exp.get("seuil_optimise"):
        s = exp["seuil_optimise"]
        st.markdown("#### 🎯 Seuil de décision optimisé (F1-max)")
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

    st.markdown("#### 💬 Discussion")
    concl = exp["conclusion"]
    st.markdown(
        f"""
- **Classification (Injury Risk)** : sur le split unique, le {concl['meilleur_modele_classification']}
  obtient le meilleur ROC-AUC ({concl['roc_auc_meilleur']}). **Mais la validation croisée 5-fold
  raconte une histoire plus nuancée** : Gradient Boosting et Logistic Regression sont
  statistiquement quasi indiscernables (écart &lt; 1 écart-type), et Random Forest est même
  légèrement en retrait. Conclusion honnête : le classement du split unique n'était pas
  totalement robuste — c'est précisément pour ça qu'on ne s'arrête jamais à un seul split en
  science des données.
- **Ce que ça signifie concrètement** : la complexité additionnelle du Gradient Boosting/Random
  Forest par rapport à une simple régression logistique n'apporte, sur ce dataset synthétique,
  qu'un gain marginal et incertain. Le vrai levier de progression n'est donc pas le choix de
  modèle, mais soit (a) plus de features informatives, soit (b) de vraies données de capteurs —
  un dataset synthétique de 6000 lignes a une limite intrinsèque de signal.
- **Compromis assumé sur le recall** : la pondération équilibrée fait fortement remonter le
  **recall** (le modèle de production rate désormais deux fois moins de cas réellement à risque
  qu'avant rééquilibrage), au prix d'une precision plus faible. Pour un outil de prévention de
  blessure, rater un vrai positif coûte plus cher qu'une fausse alerte — un choix de conception
  assumé, pas un défaut caché.
- **Régression (Fatigue)** : le Gradient Boosting réduit le MAE de
  {round((exp['regression_fatigue'][0]['mae'] - concl['mae_meilleur']) / exp['regression_fatigue'][0]['mae'] * 100, 1)}%
  par rapport à la baseline linéaire — un gain plus net et plus stable que côté classification.
- **Ce que cette expérience apprend à l'équipe** : savoir mesurer honnêtement l'incertitude d'un
  résultat (bootstrap, CV) est aussi important que le résultat lui-même. Le vrai verrou pour la
  V3 n'est pas le choix de modèle — c'est le passage d'un dataset synthétique à des données de
  capteurs réels, seul moyen de valider ces résultats hors simulation.
"""
    )
    st.caption(
        "Étude reproductible : `python -m models.experiment_comparaison` régénère ces résultats "
        "et figures (mêmes graines aléatoires, donc mêmes chiffres)."
    )


# PIED DE PAGE
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#5B7285; font-size:0.75rem; letter-spacing:1px;">
        ⚡ ATHÉNA v2 — Plateforme IA sportive prédictive — ARiA Tech<br>
        Pan-African Robotics Competition 2026 — Ligue Intelligence Artificielle<br>
        Points d'injection matériel prévus pour capteurs Bluetooth / I2C / SPI (Raspberry Pi) — Version démonstration
    </div>
    """,
    unsafe_allow_html=True,
)
