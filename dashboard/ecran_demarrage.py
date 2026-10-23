# ATHÉNA - dashboard/ecran_demarrage.py
# Écran de démarrage : logo, tracé d'ECG animé et étapes de chargement qui se cochent.
#
# Honnêteté avant tout : chaque coche correspond à une étape RÉELLEMENT terminée (imports,
# chargement des modèles, préchauffage du détecteur de pose...). Aucune attente artificielle :
# l'écran ne s'affiche qu'au tout premier chargement de l'application (ensuite tout est déjà
# en mémoire et les sessions suivantes s'ouvrent instantanément).
#
# Deux zones séparées : le fond (logo + ECG) est affiché UNE fois, pour que l'animation de
# l'ECG ne redémarre pas ; seule la liste des étapes est réécrite à chaque progression.
# Sécurité : si une erreur arrêtait le script avant la fin, l'écran s'efface tout seul après
# 30 s pour laisser voir le message d'erreur de Streamlit.

import time

import streamlit as st

ETAPES_DEMARRAGE = [
    "Moteur de calcul (NumPy, pandas, Plotly)",
    "Modèle de risque de blessure (Gradient Boosting calibré)",
    "Coach IA, moteur de décision et rapports",
    "Détecteur de posture (MediaPipe)",
    "Profil de l'athlète, risque et alertes",
    "Tableau de bord",
]

_CSS = """
<style>
@keyframes athena-ecg { from { transform: translateX(0); } to { transform: translateX(-50%); } }
@keyframes athena-pouls { 0%, 100% { opacity: .35; transform: scale(.8); } 50% { opacity: 1; transform: scale(1); } }
@keyframes athena-securite { to { opacity: 0; visibility: hidden; } }
@keyframes athena-sortie { to { opacity: 0; visibility: hidden; } }
.athena-demarrage {
  position: fixed; inset: 0; z-index: 1000000;
  background: linear-gradient(120deg, #0F2A3D 0%, #173F57 60%, #1B6E8C 100%);
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 18px;
  padding-bottom: 270px; box-sizing: border-box;   /* remonte le bloc : la liste des étapes se place dessous */
  font-family: Inter, sans-serif; color: #E3EEF3;
  animation: athena-securite .6s ease 30s forwards;
}
.athena-demarrage.sortie { animation: athena-sortie .45s ease forwards; pointer-events: none; }
.athena-demarrage img { width: 92px; height: 92px; object-fit: contain; background: #FFFFFF; padding: 8px;
  border-radius: 20px; box-shadow: 0 12px 36px rgba(0, 0, 0, .35); }
.athena-demarrage .nom { font-family: Montserrat, Inter, sans-serif; font-weight: 800; font-size: 44px;
  letter-spacing: .12em; color: #F4F7F9; line-height: 1; }
.athena-demarrage .sous { font-size: 15px; color: #B9D3DE; letter-spacing: .02em; }
.athena-ecg { width: min(560px, 86vw); height: 72px; overflow: hidden; margin: 6px 0 4px;
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 18%, #000 82%, transparent);
          mask-image: linear-gradient(90deg, transparent, #000 18%, #000 82%, transparent); }
.athena-ecg svg { width: 200%; height: 100%; animation: athena-ecg 2.4s linear infinite; }
.athena-ecg path { fill: none; stroke: #8FD0E6; stroke-width: 2.4; stroke-linejoin: round; stroke-linecap: round;
  filter: drop-shadow(0 0 4px rgba(143, 208, 230, .7)); }
.athena-etapes { position: fixed; z-index: 1000001; left: 50%; top: calc(50% + 20px); transform: translateX(-50%);
  width: min(460px, 88vw); font-family: Inter, sans-serif; color: #E3EEF3;
  animation: athena-securite .6s ease 30s forwards; }
.athena-etapes ul { list-style: none; margin: 0 0 14px; padding: 0; }
.athena-etapes li { display: flex; align-items: center; gap: 12px; font-size: 15px; padding: 4px 0; color: #6F8FA0; }
.athena-etapes li.faite { color: #DCEAF1; }
.athena-etapes li.en-cours { color: #FFFFFF; font-weight: 600; }
.athena-etapes .icone { width: 18px; height: 18px; border-radius: 50%; flex: none; border: 2px solid #3D6378;
  display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; }
.athena-etapes li.faite .icone { background: #8FD0E6; border-color: #8FD0E6; color: #0F2A3D; }
.athena-etapes li.en-cours .icone { border-color: #8FD0E6; background: #8FD0E6; animation: athena-pouls 1s ease-in-out infinite; }
.athena-barre { height: 4px; border-radius: 2px; background: rgba(255, 255, 255, .15); overflow: hidden; }
.athena-barre div { height: 100%; background: #8FD0E6; transition: width .35s ease; }
.athena-pied { margin-top: 8px; font-size: 13px; color: #9FBCCB; display: flex; justify-content: space-between; }
</style>
"""


def _trace_ecg() -> str:
    """Six battements identiques (ligne de base, onde P, complexe QRS, onde T) sur 1200 unités :
    le SVG fait 2 fois la largeur visible et glisse de moitié, la boucle est donc sans couture."""
    battement = ("l 40 0 q 8 -10 16 0 l 14 0 l 6 8 l 10 -52 l 10 64 l 6 -20 l 22 0 "
                 "q 14 -18 28 0 l 48 0")
    return ('<svg viewBox="0 0 1200 80" preserveAspectRatio="none" aria-hidden="true">'
            f'<path d="M 0 48 {battement * 6}"/></svg>')


class EcranDemarrage:
    """Affiche l'écran de démarrage ; `valider()` coche l'étape en cours, `terminer()` le fait disparaître."""

    def __init__(self, etapes: list[str] = ETAPES_DEMARRAGE):
        self.etapes = etapes
        self.faites = 0
        self.debut = time.perf_counter()
        self._fond = st.empty()
        self._liste = st.empty()
        self._fond.markdown(_CSS + self._html_fond(), unsafe_allow_html=True)
        self._afficher_liste()

    @staticmethod
    def _html_fond(classe: str = "") -> str:
        return (f'<div class="athena-demarrage {classe}">'
                '<img src="app/static/logo_athena.png" alt="Logo ATHÉNA">'
                '<div class="nom">ATHÉNA</div>'
                '<div class="sous">Plateforme IA sportive prédictive · ARiA Tech · PARC 2026</div>'
                f'<div class="athena-ecg">{_trace_ecg()}</div>'
                '</div>')

    def _afficher_liste(self):
        lignes = []
        for i, nom in enumerate(self.etapes):
            if i < self.faites:
                classe, icone = "faite", "✓"
            elif i == self.faites:
                classe, icone = "en-cours", ""
            else:
                classe, icone = "", ""
            lignes.append(f'<li class="{classe}"><span class="icone">{icone}</span>{nom}</li>')
        pct = round(100 * self.faites / len(self.etapes))
        duree = f"{time.perf_counter() - self.debut:.1f}".replace(".", ",")   # 4,2 s (virgule décimale)
        self._liste.markdown(
            f'<div class="athena-etapes"><ul>{"".join(lignes)}</ul>'
            f'<div class="athena-barre"><div style="width:{pct}%"></div></div>'
            f'<div class="athena-pied"><span>Chargement hors-ligne</span><span>{pct} % · {duree} s</span></div>'
            '</div>', unsafe_allow_html=True)

    def valider(self):
        """L'étape en cours est terminée : coche, puis passe à la suivante."""
        self.faites = min(self.faites + 1, len(self.etapes))
        self._afficher_liste()

    def terminer(self):
        """Toutes les étapes sont cochées, puis l'écran s'efface en fondu."""
        self.faites = len(self.etapes)
        self._afficher_liste()
        self._liste.empty()
        self._fond.markdown(_CSS + self._html_fond("sortie"), unsafe_allow_html=True)
