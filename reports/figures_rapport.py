# ATHÉNA - reports/figures_rapport.py
# Figures du rapport final PARC 2026, régénérées depuis les artefacts JSON (aucun chiffre saisi à la main) :
#   fig1 tâche et protocole · fig2 modèle (boosting + structure du générateur vs modèles)
#   fig3 résultats E1 · fig4 expérience E2 (profondeur) · fig5 apprentissage et robustesse · fig6 calibration et importance
#
# Lancement : python -m reports.figures_rapport   (depuis la racine du projet) -> reports/rapport_figures/

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

RACINE = Path(__file__).resolve().parent.parent
ART = RACINE / "models" / "artifacts"
SORTIE = Path(__file__).resolve().parent / "rapport_figures"
SORTIE.mkdir(exist_ok=True)

BLEU_F, BLEU, ORANGE, GRIS, VERT, ROUGE, TEAL = "#1B4B66", "#1B6E8C", "#D9822B", "#7A8794", "#2E8B57", "#B03A2E", "#4FA3A5"
LARGEUR = 6.3   # pouces ≈ largeur utile d'une page A4 avec marges de 2,54 cm

_polices = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "Carlito" if "Carlito" in _polices else "DejaVu Sans",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "mathtext.fontset": "dejavusans",
})


def fr(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def fmt_fr(nd=2):
    return FuncFormatter(lambda v, _: fr(v, nd))


def charger(nom):
    return json.loads((ART / nom).read_text(encoding="utf-8"))


def enregistrer(fig, nom):
    fig.savefig(SORTIE / nom, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _boite(ax, x, y, w, h, titre, sous, fond="#EAF2F6", bord=BLEU, taille=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fond, ec=bord, lw=1.1))
    ax.text(x + w / 2, y + h * 0.68, titre, ha="center", va="center", fontsize=taille, fontweight="bold", color=BLEU_F)
    ax.text(x + w / 2, y + h * 0.28, sous, ha="center", va="center", fontsize=taille - 1.3, color="#26343F", linespacing=1.15)


def _fleche(ax, x0, y0, x1, y1, style="-|>", couleur=BLEU_F, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=couleur, lw=1.1, linestyle=ls))


# ---------------------------------------------------------------------------------------- figure 1 ----

def fig1():
    fig, ax = plt.subplots(figsize=(LARGEUR, 2.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")
    _boite(ax, 1, 20, 21, 13, "Générateur", "6 000 profils, 10 variables\nrisque latent + bruit")
    _boite(ax, 27, 20, 20, 13, "Partition 80/20", "4 800 apprentissage\n1 200 test")
    _boite(ax, 52, 20, 21, 13, "Modèles", "LR · RF · GB · HGB\npoids équilibrés")
    _boite(ax, 78, 20, 21, 13, "Évaluation", "AUC, P, R, F1\nMAE, RMSE, R²")
    for x0, x1 in [(22.6, 26.2), (47.6, 51.2), (73.6, 77.2)]:
        _fleche(ax, x0, 26.5, x1, 26.5)
    ax.text(50, 15.4, "Contrôles complémentaires", ha="center", fontsize=8, style="italic", color=GRIS)
    _boite(ax, 1, 2, 30, 10.5, "Répétitions", "CV 5 plis · 10 partitions\n10 jeux indépendants", fond="#F4F5F6", bord=GRIS)
    _boite(ax, 35, 2, 30, 10.5, "Jeu inédit et robustesse", "graine 123 (n = 3 000)\nbruit · valeurs manquantes", fond="#F4F5F6", bord=GRIS)
    _boite(ax, 69, 2, 30, 10.5, "Plafond (oracle)", "AUC du risque latent\nsans bruit", fond="#F4F5F6", bord=GRIS)
    enregistrer(fig, "fig1_tache_protocole.png")


# ---------------------------------------------------------------------------------------- figure 2 ----

def _courbes_structure():
    """Vraie contribution d'une variable au risque latent, meilleure approximation linéaire, et forme apprise
    (dépendance partielle d'un Gradient Boosting de profondeur 1) — centrées et rapportées à l'écart-type de la vraie courbe."""
    import sys
    sys.path.insert(0, str(RACINE))
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.inspection import partial_dependence
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_sample_weight
    from models.train_injury_model import _generer_dataset

    X, y, _ = _generer_dataset(n=6000, graine=42)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=180, max_depth=1, learning_rate=0.08, random_state=42)
    gb.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    vraies = {0: lambda v: 2.6 * np.clip(v - 1.3, 0, None) ** 1.4 + 1.1 * np.clip(0.8 - v, 0, None),
              3: lambda v: 0.9 * np.clip(7.0 - v, 0, None)}
    sortie = {}
    for j, g in vraies.items():
        x = X_tr[:, j]
        lo, hi = np.percentile(x, [1, 99])
        grille = np.linspace(lo, hi, 120)
        pd_ = partial_dependence(gb, X_tr, [j], grid_resolution=80, kind="average", response_method="decision_function")
        appris = np.interp(grille, pd_["grid_values"][0], pd_["average"][0])
        a, b = np.polyfit(x, g(x), 1)
        s = g(x).std()
        c_vrai, c_lin = g(grille), a * grille + b
        c_app = appris
        ref_v, ref_l, ref_a = g(x).mean(), (a * x + b).mean(), np.interp(x, pd_["grid_values"][0], pd_["average"][0]).mean()
        sortie[j] = (grille, (c_vrai - ref_v) / s, (c_lin - ref_l) / s, (c_app - ref_a) / s)
    return sortie


def fig2():
    courbes = _courbes_structure()
    fig = plt.figure(figsize=(LARGEUR, 3.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.9], hspace=0.42, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :]); ax0.set_xlim(0, 100); ax0.set_ylim(0, 22); ax0.axis("off"); ax0.grid(False)
    boites = [(1, r"$F_0$", "log-cote de base"), (19, r"$+\,\nu\,h_1(x)$", "arbre 1"), (37, r"$+\,\nu\,h_2(x)$", "arbre 2"),
              (55, r"$\cdots$", ""), (65, r"$+\,\nu\,h_M(x)$", "arbre M")]
    for x, t, s in boites:
        w = 14 if t != r"$\cdots$" else 6
        ax0.add_patch(FancyBboxPatch((x, 8), w, 9, boxstyle="round,pad=0.3,rounding_size=1.0", fc="#EAF2F6", ec=BLEU, lw=1.1))
        ax0.text(x + w / 2, 12.5, t, ha="center", va="center", fontsize=10, color=BLEU_F)
        if s:
            ax0.text(x + w / 2, 5.2, s, ha="center", va="center", fontsize=7.5, color="#26343F")
    for x0, x1 in [(15.6, 18.0), (33.6, 36.0), (51.6, 54.0), (61.6, 64.0)]:
        _fleche(ax0, x0, 12.5, x1, 12.5)
    ax0.add_patch(FancyBboxPatch((83, 8), 16, 9, boxstyle="round,pad=0.3,rounding_size=1.0", fc="#FBEFE0", ec=ORANGE, lw=1.1))
    ax0.text(91, 14.0, r"$\sigma(F_M(x))$", ha="center", va="center", fontsize=10, color=BLEU_F)
    ax0.text(91, 10.4, "probabilité", ha="center", va="center", fontsize=7.5, color="#26343F")
    _fleche(ax0, 79.6, 12.5, 82.0, 12.5)
    ax0.text(50, 0.6, r"Réglages étudiés : profondeur $d$ des arbres (ordre d'interaction), nombre d'arbres $M$, pas $\nu$.  "
                      r"Perte : log-loss pondérée.", ha="center", fontsize=8, color=GRIS)

    titres = {0: "(a) ACWR : sous-charge et surcharge", 3: "(b) Sommeil : dette sous 7 h"}
    xlab = {0: "ACWR", 3: "Heures de sommeil"}
    for k, j in enumerate([0, 3]):
        ax = fig.add_subplot(gs[1, k])
        grille, vrai, lin, app = courbes[j]
        ax.plot(grille, vrai, color="black", lw=2.0, label="Vraie contribution (générateur)")
        ax.plot(grille, lin, color=GRIS, lw=1.6, ls="--", label="Meilleure droite (modèle linéaire)")
        ax.plot(grille, app, color=BLEU, lw=1.6, label="Appris : GB, profondeur 1")
        ax.set_title(titres[j], loc="left"); ax.set_xlabel(xlab[j])
        ax.set_ylabel("Contribution au risque (u. d'écart-type)" if k == 0 else "")
        ax.yaxis.set_major_formatter(fmt_fr(1)); ax.xaxis.set_major_formatter(fmt_fr(1))
        if k == 0:
            ax.legend(loc="upper center", frameon=False, handlelength=2.2)
    enregistrer(fig, "fig2_modele.png")


# ---------------------------------------------------------------------------------------- figure 3 ----

def _barres_valeurs(ax, xs, vals, errs, couleurs, nd, hatchs=None, decalage=0.0, y_label=None):
    for i, (x, v, e, c) in enumerate(zip(xs, vals, errs, couleurs)):
        ax.bar(x, v, 0.62, color=c, yerr=e, capsize=3, error_kw=dict(lw=1, ecolor="#26343F"),
               hatch=(hatchs[i] if hatchs else None), edgecolor="white" if not (hatchs and hatchs[i]) else "#26343F", linewidth=0.6)
        if y_label is None:
            ax.text(x, v + (e or 0) + decalage, fr(v, nd), ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#26343F")
        else:
            ax.text(x, y_label, fr(v, nd), ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#26343F",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92), zorder=5)


def fig3():
    e1, e2 = charger("experiment_results.json"), charger("experiment_profondeur.json")
    ms = e1["variance_multi_seed"]["classification_roc_auc"]
    noms = ["Rég.\nlogistique", "Random\nForest", "GB\n(réf., d = 3)", "HistGB", "GB\nd = 1 (E2)"]
    cles = ["Logistic Regression (baseline)", "Random Forest", "Gradient Boosting (retenu en production)", "HistGradientBoosting (natif NaN)"]
    p1 = e2["P1_jeu_E1_partitions"]
    vals = [ms[c]["moyenne"] for c in cles] + [p1["gb_fixe"]["1"]["moyenne"]]
    errs = [ms[c]["ecart_type"] for c in cles] + [p1["gb_fixe"]["1"]["ecart_type"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(LARGEUR, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.28))
    cols = [GRIS, ORANGE, BLEU, TEAL, BLEU]
    _barres_valeurs(a, range(5), vals, errs, cols, 3, hatchs=[None, None, None, None, "///"], y_label=0.512)
    orc = p1["oracle"]["moyenne"]
    a.axhline(orc, color="black", ls="--", lw=1.2); a.text(-0.42, orc + 0.002, "Oracle (plafond) " + fr(orc, 3), ha="left", va="bottom", fontsize=8)
    
    a.set_ylim(0.5, 0.75); a.set_xticks(range(5)); a.set_xticklabels(noms, fontsize=7.8); a.yaxis.set_major_formatter(fmt_fr(2))
    a.set_ylabel("ROC-AUC (moyenne ± écart-type, 10 partitions)"); a.set_title("(a) Risque de blessure", loc="left"); a.grid(axis="x", visible=False)

    reg = e1["regression_fatigue"]
    noms_r = ["Rég.\nlinéaire", "Random\nForest", "GB", "HistGB"]
    maes = [r["mae"] for r in reg]
    _barres_valeurs(b, range(4), maes, [0] * 4, [GRIS, ORANGE, BLEU, TEAL], 4, y_label=0.04)
    pl = e2["plancher_bruit_fatigue"]["mae"]
    b.axhline(pl, color="black", ls="--", lw=1.2); b.text(-0.42, pl + 0.015, "Plancher de bruit " + fr(pl, 4), ha="left", va="bottom", fontsize=8)
    b.set_ylim(0, 0.95); b.set_xticks(range(4)); b.set_xticklabels(noms_r, fontsize=7.8); b.yaxis.set_major_formatter(fmt_fr(1))
    b.set_ylabel("MAE (échelle 0-10, test principal)"); b.set_title("(b) Fatigue", loc="left"); b.grid(axis="x", visible=False)
    enregistrer(fig, "fig3_resultats_e1.png")


# ---------------------------------------------------------------------------------------- figure 4 ----

def fig4():
    e2 = charger("experiment_profondeur.json")
    p2 = e2["P2_jeux_independants"]
    profs = e2["protocole"]["profondeurs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(LARGEUR, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.34))
    xs = np.arange(len(profs))
    for cle, col, mk, dec, lab in [("gb_fixe", BLEU, "o", -0.07, "180 arbres"), ("gb_arret", ORANGE, "s", 0.07, "arrêt précoce")]:
        m = [p2[cle][str(d)]["moyenne"] for d in profs]
        s = [p2[cle][str(d)]["ecart_type"] for d in profs]
        a.errorbar(xs + dec, m, yerr=s, color=col, marker=mk, ms=5, lw=1.4, capsize=3, label="GB, " + lab)
    for x, d in zip(xs, profs):
        v = p2["gb_fixe"][str(d)]["moyenne"]
        a.text(x - 0.07, v - 0.0125, fr(v, 3), ha="center", va="top", fontsize=7.6, color=BLEU, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=6)
    a.axhline(p2["oracle"]["moyenne"], color="black", ls="--", lw=1.2)
    a.text(len(profs) - 0.55, p2["oracle"]["moyenne"] + 0.0015, "Oracle (plafond) " + fr(p2["oracle"]["moyenne"], 3), ha="right", va="bottom", fontsize=8)
    a.axhline(p2["lr"]["moyenne"], color=GRIS, ls=":", lw=1.4, label="Rég. logistique " + fr(p2["lr"]["moyenne"], 3))
    a.set_xticks(xs); a.set_xticklabels([str(d) for d in profs]); a.set_xlabel("Profondeur maximale des arbres d (ordre d'interaction)")
    a.set_ylabel("ROC-AUC (10 jeux indépendants)"); a.set_ylim(0.595, 0.735); a.yaxis.set_major_formatter(fmt_fr(2))
    a.set_title("(a) AUC selon la profondeur", loc="left"); a.legend(loc="lower left", frameon=False)

    ds = [d for d in profs if d != 1]
    ys = np.arange(len(ds))[::-1]
    for cle, col, mk, dec in [("gb_fixe", BLEU, "o", 0.13), ("gb_arret", ORANGE, "s", -0.13)]:
        for y, d in zip(ys, ds):
            r = p2["diff_vs_profondeur1"][cle][str(d)]
            b.errorbar(r["moyenne"], y + dec, xerr=[[r["moyenne"] - r["ic95"][0]], [r["ic95"][1] - r["moyenne"]]], color=col,
                       marker=mk, ms=5, capsize=3, lw=1.3)
            b.text(r["ic95"][0] - 0.003, y + dec, fr(r["moyenne"], 3), fontsize=6.8, color=col, va="center", ha="right")
    b.axvline(0, color=GRIS, lw=1); b.axvline(0.010, color=ROUGE, ls="--", lw=1.2)
    b.text(0.0105, ys[0] + 0.52, "seuil +0,010", color=ROUGE, fontsize=7.8, ha="left", va="center")
    b.set_yticks(ys); b.set_yticklabels([f"d = {d}" for d in ds]); b.set_ylim(-0.6, ys[0] + 0.85)
    b.set_xlim(-0.13, 0.03); b.xaxis.set_major_formatter(fmt_fr(2)); b.set_xlabel("ΔAUC par rapport à d = 1 (IC95)")
    b.set_title("(b) Gain des interactions", loc="left"); b.grid(axis="y", visible=False)
    enregistrer(fig, "fig4_profondeur.png")


# ---------------------------------------------------------------------------------------- figure 5 ----

def fig5():
    av, e2 = charger("advanced_analysis.json"), charger("experiment_profondeur.json")
    fig, axs = plt.subplots(1, 3, figsize=(LARGEUR, 2.35), gridspec_kw=dict(wspace=0.42))
    a, b, c = axs
    lc = av["courbe_apprentissage"]
    n = [r["n_train"] for r in lc]; m = np.array([r["auc_moyen"] for r in lc]); s = np.array([r["ecart_type"] for r in lc])
    a.errorbar(n, m, yerr=s, color=BLEU, marker="o", ms=4, capsize=3, lw=1.4)
    for xi, yi, si in zip(n, m, s):
        a.text(xi, yi + si + 0.004, fr(yi, 3), ha="center", va="bottom", fontsize=6.6, color=BLEU)
    a.axhline(e2["P2_jeux_independants"]["oracle"]["moyenne"], color="black", ls="--", lw=1)
    a.text(n[0], e2["P2_jeux_independants"]["oracle"]["moyenne"] + 0.004, "oracle", fontsize=7.6)
    a.set_xlabel("Exemples d'apprentissage"); a.set_ylabel("ROC-AUC"); a.set_title("(a) Apprentissage", loc="left")
    a.set_ylim(0.56, 0.74); a.yaxis.set_major_formatter(fmt_fr(2)); a.set_xticks([0, 2400, 4800])
    mq = av["robustesse"]["manquants"]
    x = [100 * r["taux_manquant"] for r in mq]
    b.plot(x, [r["production_imputation_milieu"] for r in mq], color=BLEU, marker="o", ms=4, lw=1.5, label="Production")
    for xi, r in zip(x, mq):
        b.text(xi, r["production_imputation_milieu"] + 0.006, fr(r["production_imputation_milieu"], 3), ha="center", va="bottom", fontsize=6.4, color=BLEU)
    b.plot(x, [r["hgb_natif_nan"] for r in mq], color=TEAL, marker="s", ms=4, lw=1.3, label="HistGB (NaN natif)")
    b.plot(x, [r["gb_imputation_milieu"] for r in mq], color=GRIS, marker="^", ms=4, lw=1.3, ls="--", label="GB de l'étude")
    b.set_xlabel("Valeurs manquantes (%)"); b.set_title("(b) Données manquantes", loc="left")
    b.set_ylim(0.54, 0.775); b.yaxis.set_major_formatter(fmt_fr(2)); b.legend(frameon=False, fontsize=7, loc="upper right")
    bq = av["robustesse"]["bruit"]
    c.errorbar([r["niveau_sigma"] for r in bq], [r["auc_moyen"] for r in bq], yerr=[r["ecart_type"] for r in bq],
               color=ORANGE, marker="o", ms=4, capsize=3, lw=1.4)
    for r in bq:
        if r["niveau_sigma"] in (0.1, 0.25):
            continue
        c.text(r["niveau_sigma"], r["auc_moyen"] + r["ecart_type"] + 0.004, fr(r["auc_moyen"], 3), ha="center", va="bottom", fontsize=6.6, color=ORANGE)
    c.set_xlabel("Bruit ajouté (σ, u. standardisées)"); c.set_title("(c) Bruit de mesure", loc="left")
    c.set_ylim(0.54, 0.74); c.yaxis.set_major_formatter(fmt_fr(2)); c.xaxis.set_major_formatter(fmt_fr(1))
    enregistrer(fig, "fig5_apprentissage_robustesse.png")


# ---------------------------------------------------------------------------------------- figure 6 ----

LIBELLES = {"acwr": "ACWR", "charge_chronique": "Charge chronique", "fc_repos_estimee": "FC de repos", "sommeil_h": "Sommeil",
            "age": "Âge", "imc": "IMC", "historique_blessures": "Blessures passées", "deficit_calorique_pct": "Déficit calorique",
            "lactate": "Lactate", "intensite_moyenne_7j": "Intensité 7 j"}


def fig6():
    av, e2 = charger("advanced_analysis.json"), charger("experiment_profondeur.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(LARGEUR, 2.7), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.5))
    cl = av["calibration_production"]["fiabilite_par_classe"]
    a.plot([0, 0.8], [0, 0.8], color=GRIS, ls="--", lw=1, label="Calibration parfaite")
    a.plot([r["score_moyen"] for r in cl], [r["frequence_observee"] for r in cl], color=BLEU, marker="o", ms=4.5, lw=1.4, label="Score brut")
    a.plot([r["prob_corrigee_moyenne"] for r in cl], [r["frequence_observee"] for r in cl], color=ORANGE, marker="s", ms=4.5, lw=1.4,
           label="Après correction de prévalence")
    a.set_xlabel("Probabilité prédite"); a.set_ylabel("Fréquence observée de blessure"); a.set_title("(a) Fiabilité des probabilités", loc="left")
    a.set_xlim(0, 0.8); a.set_ylim(0, 0.8); a.xaxis.set_major_formatter(fmt_fr(1)); a.yaxis.set_major_formatter(fmt_fr(1))
    a.legend(frameon=False, fontsize=7.3, loc="upper left")
    iv = e2["importance_vs_verite"]
    ordre = np.argsort(iv["baisse_auc_oracle"])
    y = np.arange(len(ordre))
    b.barh(y + 0.19, [iv["baisse_auc_oracle"][i] for i in ordre], 0.36, color="black", label="Oracle (vérité du générateur)")
    b.barh(y - 0.19, [iv["baisse_auc_modele"][i] for i in ordre], 0.36, color=BLEU, label="Modèle GB")
    for yy, i in zip(y, ordre):
        for off, v in ((0.19, iv["baisse_auc_oracle"][i]), (-0.19, iv["baisse_auc_modele"][i])):
            if v > 0.004:
                b.text(v + 0.0012, yy + off, fr(v, 3), va="center", ha="left", fontsize=6.3)
    b.set_yticks(y); b.set_yticklabels([LIBELLES[iv["variables"][i]] for i in ordre], fontsize=7.8)
    b.set_xlabel("Baisse d'AUC si la variable est permutée"); b.xaxis.set_major_formatter(fmt_fr(2)); b.set_xlim(-0.008, 0.092)
    b.set_title("(b) Importance (ρ de Spearman = " + fr(iv["spearman_rho"], 2) + ")", loc="left"); b.legend(frameon=False, fontsize=7.3, loc="lower right"); b.grid(axis="y", visible=False)
    enregistrer(fig, "fig6_calibration_importance.png")


# ---------------------------------------------------------------------------------------- figure 7 ----

def fig7():
    e3 = charger("experiment_interaction.json")
    forces = e3["protocole"]["forces"]; profs = e3["protocole"]["profondeurs"]
    res = e3["resultats"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(LARGEUR, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.34))
    cols = [GRIS, TEAL, ORANGE, BLEU]
    xs = np.arange(len(profs))
    for c, col in zip(forces, cols):
        m = [res[str(c)]["agrege"]["gb_fixe"][str(d)]["moyenne"] for d in profs]
        a.plot(xs, m, color=col, marker="o", ms=4.5, lw=1.5, label="c = " + fr(c, 2 if c in (0.25,) else 1))
        a.axhline(res[str(c)]["agrege"]["oracle"]["moyenne"], color=col, ls=":", lw=0.9)
        im = int(np.argmax(m))
        if c >= 0.5:
            a.text(xs[im], m[im] + 0.004, fr(m[im], 3), ha="center", va="bottom", fontsize=6.8, color=col)
    a.set_xticks(xs); a.set_xticklabels([str(d) for d in profs]); a.set_xlabel("Profondeur maximale des arbres d")
    a.set_ylabel("ROC-AUC (10 jeux, 180 arbres)"); a.yaxis.set_major_formatter(fmt_fr(2)); a.set_ylim(0.60, 0.79)
    a.set_title("(a) AUC selon d et la force c", loc="left"); a.legend(frameon=False, fontsize=7, loc="upper left", ncol=2, title="interaction", title_fontsize=7)

    h7 = e3["verdicts"]["H7"]["regrets"]
    larg = 0.26
    for k, (p, col) in enumerate([("1", BLEU), ("2", ORANGE), ("3", TEAL)]):
        vals = [h7[str(c)][p] for c in forces]
        bars = b.bar(np.arange(len(forces)) + (k - 1) * larg, vals, larg, color=col, label="d = " + p)
        for x, v in zip(np.arange(len(forces)) + (k - 1) * larg, vals):
            if v > 0.0005:
                b.text(x, v + 0.0012, fr(v, 3), ha="center", va="bottom", fontsize=5.8, rotation=90)
    b.axhline(0.012, color=ROUGE, ls="--", lw=1.1); 
    b.set_xticks(np.arange(len(forces))); b.set_xticklabels([fr(c, 2) for c in forces]); b.set_xlabel("Force de l'interaction c")
    b.set_ylabel("Regret d'AUC (meilleur d − d)"); b.yaxis.set_major_formatter(fmt_fr(2)); b.set_ylim(0, 0.058)
    b.set_title("(b) Regret de chaque profondeur", loc="left"); b.legend(frameon=False, fontsize=7, loc="upper left"); b.grid(axis="x", visible=False)
    enregistrer(fig, "fig7_interaction.png")


def main():
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
    print("Figures écrites dans", SORTIE)


if __name__ == "__main__":
    main()
