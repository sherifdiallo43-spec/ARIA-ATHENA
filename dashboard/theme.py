# ATHÉNA - dashboard/theme.py
# Identité visuelle du tableau de bord : palette, CSS, modèle Plotly, en-tête.
# Tout le style est regroupé ici pour que app.py ne contienne que la logique.
#
# Polices : Inter (texte, chiffres) et Montserrat (titres), hébergées dans
# dashboard/static/fonts et déclarées dans .streamlit/config.toml : aucun
# chargement réseau, la démo marche hors-ligne.
# Choix de conception : thème clair à fort contraste, pensé pour la projection
# en salle (un fond sombre devient gris délavé sur un vidéoprojecteur). Une
# seule couleur d'accent (bleu pétrole) ; le vert / ambre / rouge sont réservés
# aux niveaux de risque, pour que la couleur porte toujours une information.

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PALETTE = {
    "encre": "#0F2A3D",        # titres, en-tête
    "texte": "#1F2937",
    "texte_doux": "#5B7285",
    "accent": "#1B6E8C",       # couleur principale de l'interface et des graphiques
    "accent_clair": "#5B9BB0",
    "fond": "#F4F7F9",
    "carte": "#FFFFFF",
    "bordure": "#E2E8F0",
    # Niveaux de risque (teintes foncées : lisibles en texte sur fond blanc)
    "ok": "#1F7A52",
    "attention": "#B26B00",
    "critique": "#C0283D",
    "info": "#1B6E8C",
}

# Séquence de couleurs des graphiques : accent d'abord, puis teintes distinctes.
COULEURS_SERIES = ["#1B6E8C", "#E07B39", "#0F2A3D", "#5B9BB0", "#8E6CC0", "#7A8B99"]

# Modèle Plotly appliqué par-dessus "plotly_white" : police, couleurs, marges.
pio.templates["athena"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, sans-serif", size=13, color=PALETTE["texte"]),
        title=dict(font=dict(size=15, color=PALETTE["encre"]), x=0, xanchor="left"),
        colorway=COULEURS_SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#EDF1F4", linecolor=PALETTE["bordure"], zeroline=False),
        yaxis=dict(gridcolor="#EDF1F4", linecolor=PALETTE["bordure"], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(font=dict(family="Inter, sans-serif")),
    )
)
MODELE_PLOTLY = "plotly_white+athena"


def classe_niveau(niveau: str) -> str:
    """Classe CSS de pastille pour un niveau de risque ou d'alerte."""
    return {
        "Faible": "pastille-ok", "OK": "pastille-ok",
        "Modéré": "pastille-attention", "Attention": "pastille-attention",
        "Élevé": "pastille-critique", "Critique": "pastille-critique",
    }.get(niveau, "pastille-info")


def injecter_theme():
    p = PALETTE
    st.markdown(
        f"""
        <style>
        /* --- Base --- */
        .stApp {{ background: {p['fond']}; color: {p['texte']}; }}
        .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }}
        h1, h2, h3, h4 {{ color: {p['encre']} !important; font-weight: 700 !important; letter-spacing: -0.01em; }}
        h3 {{ font-size: 1.25rem !important; margin-top: 0.4rem; }}
        h4 {{ font-size: 1.02rem !important; margin-top: 1.1rem; padding-bottom: 6px;
              border-bottom: 2px solid {p['bordure']}; }}
        [data-testid="stCaptionContainer"] {{ color: {p['texte_doux']}; }}

        /* --- Masquer l'habillage de développement de Streamlit --- */
        /* On ne masque PAS stToolbar : il contient le bouton qui rouvre la barre latérale repliée. */
        [data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
        [data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stExpandSidebarButton"] {{
            display: flex !important; visibility: visible !important;
            background: {p['carte']}; border: 1px solid {p['bordure']}; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12);
        }}

        /* --- Barre latérale --- */
        section[data-testid="stSidebar"] {{ background: {p['carte']}; border-right: 1px solid {p['bordure']}; }}
        section[data-testid="stSidebar"] h3 {{
            font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.07em;
            color: {p['texte_doux']} !important; margin-top: 0.6rem;
        }}
        .marque {{ padding: 4px 0 14px 0; border-bottom: 1px solid {p['bordure']}; margin-bottom: 6px; }}
        .marque-nom {{ font-family: Montserrat, Inter, sans-serif; font-size: 1.45rem; font-weight: 800; color: {p['encre']}; letter-spacing: 0.04em; }}
        .marque-nom span {{ color: {p['accent']}; }}
        .marque-logo {{ display: block; width: 100%; max-width: 210px; height: auto; margin: 0 auto 4px auto; }}
        .marque-sous {{ text-align: center; }}
        .entete-gauche {{ display: flex; align-items: center; gap: 20px; }}
        .entete-logo {{
            background: #FFFFFF; border-radius: 12px; padding: 6px 10px; height: 104px; width: auto;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18); flex-shrink: 0;
        }}
        .marque-sous {{ font-size: 0.72rem; color: {p['texte_doux']}; letter-spacing: 0.1em; text-transform: uppercase; }}

        /* --- En-tête principal --- */
        .entete {{
            background: linear-gradient(120deg, {p['encre']} 0%, #173F57 60%, {p['accent']} 100%);
            border-radius: 14px; padding: 20px 26px; margin-bottom: 14px; color: #FFFFFF;
            display: flex; justify-content: space-between; align-items: center; gap: 18px; flex-wrap: wrap;
        }}
        .entete-titre {{ font-family: Montserrat, Inter, sans-serif; font-size: 1.5rem; font-weight: 800; letter-spacing: 0.02em; line-height: 1.2; }}
        .entete-sous {{ font-size: 0.85rem; opacity: 0.82; margin-top: 4px; }}
        .entete-droite {{ display: flex; gap: 10px; align-items: stretch; flex-wrap: wrap; }}
        .entete-bloc {{
            background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.18);
            border-radius: 10px; padding: 8px 14px; min-width: 120px;
        }}
        .entete-bloc-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.75; }}
        .entete-bloc-valeur {{ font-size: 1.05rem; font-weight: 700; margin-top: 2px; }}

        /* --- Onglets --- */
        div[data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {p['bordure']}; }}
        button[data-baseweb="tab"] {{ color: {p['texte_doux']} !important; font-weight: 600; padding: 8px 9px; }}
        button[data-baseweb="tab"] p {{ font-size: 0.86rem; }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color: {p['accent']} !important; }}
        div[data-baseweb="tab-highlight"] {{ background-color: {p['accent']} !important; height: 3px; }}

        /* --- Métriques natives --- */
        div[data-testid="stMetric"] {{
            background: {p['carte']}; border: 1px solid {p['bordure']}; border-left: 4px solid {p['accent']};
            border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }}
        div[data-testid="stMetricLabel"] {{ color: {p['texte_doux']} !important; font-weight: 600; }}
        div[data-testid="stMetricLabel"] p {{ font-size: 0.82rem !important; }}
        div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {{
            color: {p['encre']} !important; font-weight: 700; font-size: 1.5rem !important;
            line-height: 1.3; font-variant-numeric: tabular-nums;
        }}

        /* --- Monitoring : constantes vitales compactes --- */
        .st-key-vitaux div[data-testid="stMetric"] {{ padding: 8px 12px; }}
        .st-key-vitaux div[data-testid="stMetricLabel"] p {{ font-size: 0.74rem !important; }}
        .st-key-vitaux div[data-testid="stMetricValue"],
        .st-key-vitaux div[data-testid="stMetricValue"] > div {{ font-size: 1.15rem !important; line-height: 1.25; }}

        /* --- Boutons --- */
        .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {{
            background: {p['accent']}; color: #FFFFFF; border: none; border-radius: 8px; font-weight: 600;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover {{
            background: #15586F; color: #FFFFFF;
        }}
        .stButton>button:disabled {{ background: #C9D5DD; color: #FFFFFF; }}

        /* --- Cartes KPI --- */
        .kpi-card {{
            background: {p['carte']}; border: 1px solid {p['bordure']}; border-radius: 14px;
            padding: 18px 16px; text-align: center; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
            border-top: 4px solid var(--kpi-couleur, {p['accent']});
        }}
        .kpi-value {{ font-size: 1.9rem; font-weight: 800; font-variant-numeric: tabular-nums; color: var(--kpi-couleur, {p['encre']}); line-height: 1.1; }}
        .kpi-label {{ color: {p['texte_doux']}; font-size: 0.74rem; letter-spacing: 0.08em;
                      text-transform: uppercase; font-weight: 600; margin-top: 6px; }}

        /* --- Pastilles de niveau --- */
        .pastille-ok, .pastille-attention, .pastille-critique, .pastille-info {{
            display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
            font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #FFFFFF;
        }}
        .pastille-ok {{ background: {p['ok']}; }}
        .pastille-attention {{ background: {p['attention']}; }}
        .pastille-critique {{ background: {p['critique']}; }}
        .pastille-info {{ background: {p['info']}; }}

        /* --- Cartes d'alerte et de recommandation --- */
        .carte-alerte {{
            background: {p['carte']}; border: 1px solid {p['bordure']}; border-left: 4px solid var(--alerte-couleur, {p['accent']});
            border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }}
        .carte-alerte p {{ margin: 6px 0 0 0; color: #334155; }}

        /* --- Pied de page --- */
        .pied {{ text-align: center; color: {p['texte_doux']}; font-size: 0.74rem; letter-spacing: 0.04em;
                 border-top: 1px solid {p['bordure']}; padding-top: 18px; margin-top: 28px; line-height: 1.7; }}
        .pied-partenaire {{ display: flex; flex-direction: column; align-items: center; gap: 8px; margin-bottom: 14px; }}
        .pied-partenaire-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 600; }}
        .pied-partenaire img {{
            height: 110px; width: auto; background: #FFFFFF; border: 1px solid {p['bordure']};
            border-radius: 12px; padding: 6px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def couleur_niveau(niveau: str) -> str:
    return PALETTE[{"pastille-ok": "ok", "pastille-attention": "attention",
                    "pastille-critique": "critique"}.get(classe_niveau(niveau), "info")]


def marque_barre_laterale():
    st.markdown(
        '<div class="marque"><img class="marque-logo" src="app/static/logo_athena.png" '
        'alt="ATHÉNA, Advanced Total Health and Energy Network Artificial Intelligence">'
        '<div class="marque-sous">ARiA Tech · PARC 2026</div></div>',
        unsafe_allow_html=True,
    )


def en_tete(athlete: str, profil_risque: dict, acwr_info: dict, mode_simulation: bool,
            source: str | None = None):
    """Bandeau supérieur : identité du produit + statut de l'athlète en un coup d'œil."""
    cat = profil_risque["injury_risk_categorie"]
    source = source or ("Simulation" if mode_simulation else "Capteurs")
    st.markdown(
        f"""
        <div class="entete">
          <div class="entete-gauche">
            <img class="entete-logo" src="app/static/logo_athena.png" alt="Logo ATHÉNA">
            <div>
              <div class="entete-titre">Plateforme IA sportive prédictive</div>
              <div class="entete-sous">Risque de blessure, fatigue et charge d'entraînement · ARiA Tech · PARC 2026, Ligue IA</div>
            </div>
          </div>
          <div class="entete-droite">
            <div class="entete-bloc"><div class="entete-bloc-label">Athlète</div>
              <div class="entete-bloc-valeur">{athlete}</div></div>
            <div class="entete-bloc"><div class="entete-bloc-label">Risque de blessure</div>
              <div class="entete-bloc-valeur">{profil_risque['injury_risk_pct']} %
                <span class="{classe_niveau(cat['niveau'])}" style="margin-left:6px;">{cat['niveau']}</span></div></div>
            <div class="entete-bloc"><div class="entete-bloc-label">ACWR</div>
              <div class="entete-bloc-valeur">{acwr_info['acwr']} · {acwr_info['zone']}</div></div>
            <div class="entete-bloc"><div class="entete-bloc-label">Source</div>
              <div class="entete-bloc-valeur">{source}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def carte_kpi(valeur: str, libelle: str, niveau: str):
    couleur = couleur_niveau(niveau)
    st.markdown(
        f"""<div class="kpi-card" style="--kpi-couleur:{couleur};">
        <div class="kpi-value">{valeur}</div>
        <div class="kpi-label">{libelle}</div>
        <div style="margin-top:8px;"><span class="{classe_niveau(niveau)}">{niveau}</span></div></div>""",
        unsafe_allow_html=True,
    )


def carte_alerte(niveau: str, titre: str, texte: str):
    st.markdown(
        f"""<div class="carte-alerte" style="--alerte-couleur:{couleur_niveau(niveau)};">
        <span class="{classe_niveau(niveau)}">{niveau}</span>
        <strong style="margin-left:10px;">{titre}</strong><p>{texte}</p></div>""",
        unsafe_allow_html=True,
    )


def pied_de_page():
    # Logo servi en local par Streamlit (dashboard/static, server.enableStaticServing).
    st.markdown(
        """<div class="pied">
        <div class="pied-partenaire">
          <div class="pied-partenaire-label">Partenaire</div>
          <img src="app/static/logo_guindy.png" alt="Guindy Technology, Smart IT Services">
        </div>
        ATHÉNA v2 · Plateforme IA sportive prédictive · ARiA Tech<br>
        Pan-African Robotics Competition 2026 · Ligue Intelligence Artificielle<br>
        Prototype d'aide à la décision sur données simulées, sans prétention diagnostique</div>""",
        unsafe_allow_html=True,
    )
