# ATHÉNA - reports/pdf_report.py
# Génération automatique d'un rapport PDF exportable : état de l'athlète,
# risques détectés, recommandations, actions prioritaires.

from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image, PageBreak
)
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader

TEAL = colors.HexColor("#0B7A6B")
TEAL_DARK = colors.HexColor("#062E2A")
GRIS = colors.HexColor("#4A4A4A")
ROUGE = colors.HexColor("#C62839")
ORANGE = colors.HexColor("#C97A0C")
VERT = colors.HexColor("#0E8A5F")

_COULEUR_NIVEAU = {"Faible": VERT, "Modéré": ORANGE, "Élevé": ROUGE}


def _styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(name="AthenaTitre", fontSize=20, leading=24,
                             textColor=TEAL_DARK, spaceAfter=4, fontName="Helvetica-Bold"))
    base.add(ParagraphStyle(name="AthenaSousTitre", fontSize=10.5, leading=14,
                             textColor=GRIS, spaceAfter=14))
    base.add(ParagraphStyle(name="AthenaSection", fontSize=13, leading=16,
                             textColor=TEAL_DARK, spaceBefore=14, spaceAfter=8, fontName="Helvetica-Bold"))
    base.add(ParagraphStyle(name="AthenaCorps", fontSize=10, leading=14, textColor=colors.HexColor("#222222")))
    base.add(ParagraphStyle(name="AthenaFootnote", fontSize=7.5, leading=10, textColor=colors.grey, alignment=TA_CENTER))
    return base


def generer_rapport_pdf(chemin_sortie: str, athlete: dict, profil_risque: dict,
                         acwr_info: dict, performance: dict, recommandations: list,
                         alertes: list, etude_scientifique: dict | None = None,
                         explication: list | None = None, plan_action: dict | None = None) -> str:
    """
    Génère le rapport PDF complet ("Generate AI Report") et l'enregistre au
    chemin indiqué. Si `etude_scientifique` est fourni (voir
    `models.experiment_comparaison`), une section scientifique (Research
    Question / Hypothèse / Results / Figures / Discussion) est ajoutée sur une
    page dédiée. Retourne le chemin du fichier généré.
    """
    styles = _styles()
    doc = SimpleDocTemplate(
        chemin_sortie, pagesize=A4,
        topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    elements = []

    # --- En-tête ---
    elements.append(Paragraph("ARiA TECH — ATHÉNA", styles["AthenaTitre"]))
    elements.append(Paragraph(
        f"Rapport IA de performance & santé — {athlete.get('nom', 'Athlète')} · "
        f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} · PARC 2026, Ligue Intelligence Artificielle",
        styles["AthenaSousTitre"],
    ))
    elements.append(HRFlowable(width="100%", color=TEAL, thickness=1.2))
    elements.append(Spacer(1, 10))

    # --- Résumé exécutif ---
    elements.append(Paragraph("État de l'athlète — Résumé exécutif", styles["AthenaSection"]))
    niveau_risque = profil_risque["injury_risk_categorie"]["niveau"]
    couleur_risque = _COULEUR_NIVEAU.get(niveau_risque, GRIS)

    data_resume = [
        ["Indicateur", "Valeur", "Statut"],
        ["Injury Risk Score", f"{profil_risque['injury_risk_pct']} %", niveau_risque],
        ["Fatigue Level", f"{profil_risque['fatigue_score']} / 10", profil_risque["fatigue_categorie"]["niveau"]],
        ["Overtraining Risk", f"{profil_risque['overtraining_pct']} %", profil_risque["overtraining_categorie"]["niveau"]],
        ["ACWR (charge)", f"{acwr_info['acwr']}", acwr_info["zone"]],
        ["Athlete Performance Score", f"{performance['score_global']} / 100", ""],
    ]
    table_resume = Table(data_resume, colWidths=[70 * mm, 40 * mm, 55 * mm])
    table_resume.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F6")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TEXTCOLOR", (2, 1), (2, 1), couleur_risque),
        ("FONTNAME", (2, 1), (2, 1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table_resume)
    elements.append(Spacer(1, 12))

    # --- Facteurs de risque contributifs ---
    if profil_risque.get("facteurs_cles"):
        elements.append(Paragraph("Facteurs contributifs principaux (modèle ML)", styles["AthenaSection"]))
        data_facteurs = [["Facteur", "Poids relatif", "Valeur observée"]]
        for f in profil_risque["facteurs_cles"]:
            data_facteurs.append([f["facteur"], f"{f['poids_relatif']:.2f}", f"{f['valeur_observee']}"])
        table_facteurs = Table(data_facteurs, colWidths=[85 * mm, 40 * mm, 40 * mm])
        table_facteurs.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E8A5F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F6")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(table_facteurs)
        elements.append(Spacer(1, 12))

    # --- Explication individuelle + plan d'action chiffré ---
    if explication:
        elements.append(Paragraph("Pourquoi ce score ? — Contributions individuelles", styles["AthenaSection"]))
        data_ex = [["Facteur", "Valeur", "Contribution au risque"]] + [
            [e["facteur"], f"{e['valeur']}", f"{e['contribution_pts']:+.1f} pts"] for e in explication
        ]
        t_ex = Table(data_ex, colWidths=[85 * mm, 30 * mm, 50 * mm])
        t_ex.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL_DARK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F6")]),
        ]))
        elements.append(t_ex)
        elements.append(Spacer(1, 10))
    if plan_action and plan_action.get("leviers"):
        elements.append(Paragraph("Plan d'action chiffré", styles["AthenaSection"]))
        for lv in plan_action["leviers"]:
            elements.append(Paragraph(
                f"<b>{lv['action']}</b> ({lv['de']} -&gt; {lv['vers']}) : -{lv['gain_pts']} pts de risque",
                styles["AthenaCorps"]))
            elements.append(Spacer(1, 3))
        pc = plan_action["plan_combine"]
        elements.append(Paragraph(
            f"<b>Plan combiné :</b> risque {pc['avant_pct']} % -&gt; {pc['apres_pct']} % (-{pc['gain_pts']} pts).",
            styles["AthenaCorps"]))
        elements.append(Spacer(1, 12))

    # --- Alertes détectées ---
    elements.append(Paragraph("Risques et alertes détectés", styles["AthenaSection"]))
    for niveau, titre, detail in alertes[:6]:
        couleur = {"Critique": ROUGE, "Attention": ORANGE, "Info": TEAL, "OK": VERT}.get(niveau, GRIS)
        elements.append(Paragraph(
            f'<font color="{couleur.hexval()}"><b>[{niveau}]</b></font> {titre} — {detail}',
            styles["AthenaCorps"],
        ))
        elements.append(Spacer(1, 3))
    elements.append(Spacer(1, 8))

    # --- Recommandations IA priorisées ---
    elements.append(Paragraph("Recommandations IA — Actions prioritaires", styles["AthenaSection"]))
    reco_triees = sorted(recommandations, key=lambda r: r["priorite"])
    for i, r in enumerate(reco_triees, start=1):
        elements.append(Paragraph(
            f"<b>{i}. [{r['categorie']}]</b> {r['texte']}", styles["AthenaCorps"]
        ))
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", color=TEAL, thickness=0.8))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "Rapport généré automatiquement par ATHÉNA (ARiA Tech) à partir de modèles de Machine Learning "
        "entraînés sur des données synthétiques calibrées sur la littérature sportive, en l'attente du "
        "déploiement de capteurs physiques. Document d'aide à la décision — ne remplace pas un avis médical.",
        styles["AthenaFootnote"],
    ))

    # --- Étude scientifique (Research Question / Hypothèse / Results / Figures) ---
    if etude_scientifique:
        elements.append(PageBreak())
        elements.append(Paragraph("ARiA TECH — ATHÉNA", styles["AthenaTitre"]))
        elements.append(Paragraph("Annexe — Étude scientifique comparative", styles["AthenaSousTitre"]))
        elements.append(HRFlowable(width="100%", color=TEAL, thickness=1.2))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Référence scientifique", styles["AthenaSection"]))
        elements.append(Paragraph(etude_scientifique.get("reference_scientifique", ""), styles["AthenaCorps"]))
        elements.append(Spacer(1, 8))

        elements.append(Paragraph("Research Question", styles["AthenaSection"]))
        elements.append(Paragraph(etude_scientifique["research_question"], styles["AthenaCorps"]))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("Hypothèse", styles["AthenaSection"]))
        elements.append(Paragraph(f"<b>H1</b> — {etude_scientifique['hypothese_h1']}", styles["AthenaCorps"]))
        elements.append(Paragraph(f"<b>H0</b> — {etude_scientifique['hypothese_h0']}", styles["AthenaCorps"]))
        th = etude_scientifique["test_hypothese"]
        elements.append(Paragraph(
            f"Test (bootstrap apparié, {th['n_tirages_valides']} tirages) : "
            f"P(AUC Gradient Boosting &gt; AUC Logistic Regression) = {th['proportion_gb_superieur'] * 100:.1f}% · "
            f"écart moyen AUC = {th['diff_auc_moyenne']:+.3f} · IC95% = "
            f"[{th['ic95_diff_auc'][0]:+.3f}, {th['ic95_diff_auc'][1]:+.3f}].",
            styles["AthenaCorps"],
        ))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Results — Classification (Injury Risk)", styles["AthenaSection"]))
        entetes_clf = ["Modèle", "Acc.", "Précision", "Recall", "F1", "ROC-AUC"]
        data_clf = [entetes_clf] + [
            [r["modele"], f"{r['accuracy']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}",
             f"{r['f1']:.3f}", f"{r['roc_auc']:.3f}"]
            for r in etude_scientifique["classification_injury_risk"]
        ]
        table_clf = Table(data_clf, colWidths=[55 * mm, 20 * mm, 24 * mm, 20 * mm, 18 * mm, 22 * mm])
        table_clf.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F6")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(table_clf)
        elements.append(Spacer(1, 12))

        cv = etude_scientifique.get("validation_croisee_5fold")
        if cv:
            elements.append(Paragraph("Robustesse — Validation croisée stratifiée (5-fold)", styles["AthenaSection"]))
            data_cv = [["Modèle", "ROC-AUC (moy. ± é.t.)", "F1 (moy. ± é.t.)"]] + [
                [r["modele"], f"{r['roc_auc_moyen']:.3f} ± {r['roc_auc_ecart_type']:.3f}",
                 f"{r['f1_moyen']:.3f} ± {r['f1_ecart_type']:.3f}"]
                for r in cv
            ]
            table_cv = Table(data_cv, colWidths=[65 * mm, 55 * mm, 55 * mm])
            table_cv.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TEAL_DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F6")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(table_cv)
            elements.append(Spacer(1, 10))

        seuil = etude_scientifique.get("seuil_optimise")
        if seuil:
            elements.append(Paragraph(
                f"Seuil de décision optimisé (F1-max) pour Gradient Boosting : {seuil['seuil_optimal']:.3f} "
                f": Accuracy={seuil['accuracy']:.3f}, Precision={seuil['precision']:.3f}, "
                f"Recall={seuil['recall']:.3f}, F1={seuil['f1']:.3f}.",
                styles["AthenaCorps"],
            ))
            elements.append(Spacer(1, 10))

        chemins_figures = etude_scientifique.get("figures", {})
        for cle, largeur in [("roc_curves", 90), ("confusion_matrix", 70), ("feature_importance", 90)]:
            chemin_img = chemins_figures.get(cle)
            if chemin_img and Path(chemin_img).exists():
                # Hauteur calculée depuis les dimensions réelles : pas de déformation.
                w_px, h_px = ImageReader(chemin_img).getSize()
                elements.append(Image(chemin_img, width=largeur * mm, height=largeur * mm * h_px / w_px))
                elements.append(Spacer(1, 8))

        elements.append(Paragraph("Discussion", styles["AthenaSection"]))
        concl = etude_scientifique["conclusion"]
        elements.append(Paragraph(
            f"Meilleur modèle classification (split unique) : {concl['meilleur_modele_classification']} "
            f"(ROC-AUC = {concl['roc_auc_meilleur']}). En validation croisée 5-fold, Gradient Boosting et "
            "Logistic Regression sont statistiquement quasi indiscernables — le classement du split "
            "unique n'était pas totalement robuste, ce qui souligne l'intérêt de ne jamais conclure sur "
            "un seul split. Meilleur modèle régression fatigue : "
            f"{concl['meilleur_modele_regression']} (MAE = {concl['mae_meilleur']}). Le recall a été "
            "priorisé via une pondération équilibrée des classes : pour un outil de prévention, "
            "manquer un athlète réellement à risque coûte plus cher qu'une fausse alerte.",
            styles["AthenaCorps"],
        ))

    doc.build(elements)
    return chemin_sortie
