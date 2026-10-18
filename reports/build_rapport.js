// Générateur du rapport final ATHÉNA (docx). Les tableaux sont lus dans les JSON du projet.
const fs = require('fs');
const path = require('path');
const D = require('docx');
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, BorderStyle, Footer, PageNumber, PageBreak, HeadingLevel } = D;

const ROOT = process.argv[2];            // .../v10/athena
const OUT = process.argv[3];
const ART = path.join(ROOT, 'models/artifacts');
const FIG = path.join(ROOT, 'reports/rapport_figures');
const J = f => JSON.parse(fs.readFileSync(path.join(ART, f), 'utf8'));
const e1 = J('experiment_results.json'), e2 = J('experiment_profondeur.json'), av = J('advanced_analysis.json');
const p2 = e2.P2_jeux_independants, p1 = e2.P1_jeu_E1_partitions;
const e3 = J('experiment_interaction.json'), R3 = e3.resultats, V3 = e3.verdicts;

const fr = (x, nd = 3) => Number(Number(x) + Math.sign(x) * 1e-9).toFixed(nd).replace('.', ',').replace('-', '−');
const frs = (x, nd = 3) => (x > 0 ? '+' : '') + fr(x, nd);
const ic = a => '[' + fr(a[0]) + ' ; ' + fr(a[1]) + ']';

const FONT = 'Calibri', BODY = 22;       // 11 pt
const BLUE = '1F5F7A';

// ---- mini-balisage : **gras**, //italique//, _{indice}, ^{exposant}
function runs(text, base = {}) {
  text = text.replace(/https:\/\//g, 'https:/\u200b/');
  const out = []; let bold = false, ital = false, i = 0, buf = '';
  const flush = (extra = {}) => { if (buf) { out.push(new TextRun({ text: buf, font: FONT, size: BODY, ...base, bold: bold || base.bold, italics: ital || base.italics, ...extra })); buf = ''; } };
  while (i < text.length) {
    if (text.startsWith('**', i)) { flush(); bold = !bold; i += 2; }
    else if (text.startsWith('//', i)) { flush(); ital = !ital; i += 2; }
    else if (text.startsWith('_{', i) || text.startsWith('^{', i)) {
      const sub = text[i] === '_'; const j = text.indexOf('}', i); flush();
      buf = text.slice(i + 2, j); flush(sub ? { subScript: true } : { superScript: true }); i = j + 1;
    } else { buf += text[i]; i++; }
  }
  flush(); return out;
}
const P = (text, o = {}) => new Paragraph({
  children: runs(text, o.run || {}), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { before: o.before ?? 0, after: o.after ?? 70, line: o.line ?? 245 }, keepNext: o.keepNext, indent: o.indent,
});
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 150, after: 70 },
  children: [new TextRun({ text: t, font: FONT, size: 27, bold: true, color: BLUE })],
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 1 } } });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 90, after: 40 },
  children: [new TextRun({ text: t, font: FONT, size: 23, bold: true, color: '26343F' })] });
const CAP = t => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 20, after: 90, line: 230 },
  children: runs(t, { size: 19 }) });
const EQ = t => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 20, after: 70 }, children: runs(t, { italics: true }) });

function IMG(file, widthIn = 6.27) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = Math.round(widthIn * 96), hpx = Math.round(wpx * h / w);
  return new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 40, after: 0 },
    children: [new ImageRun({ type: 'png', data: buf, transformation: { width: wpx, height: hpx },
      altText: { title: file, description: file, name: file } })] });
}

// ---- tableaux
function TABLE(head, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const bd = { style: BorderStyle.SINGLE, size: 4, color: 'B8C4CC' };
  const borders = { top: bd, bottom: bd, left: bd, right: bd };
  const cell = (t, w, hd, al, bold, kn) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: hd ? { type: ShadingType.CLEAR, fill: 'DCE9EF', color: 'auto' } : (o.shade && o.shade(t) ? { type: ShadingType.CLEAR, fill: 'F1F7E9', color: 'auto' } : undefined),
    margins: { top: 30, bottom: 30, left: 70, right: 70 },
    children: [new Paragraph({ alignment: al, keepNext: kn, spacing: { before: 0, after: 0, line: 228 },
      children: runs(String(t), { size: 19, bold: hd || bold }) })],
  });
  const al = i => (i === 0 ? AlignmentType.LEFT : (o.left && o.left.includes(i) ? AlignmentType.LEFT : AlignmentType.CENTER));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [new TableRow({ tableHeader: true, cantSplit: true, children: head.map((h, i) => cell(h, widths[i], true, al(i), false, true)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], false, al(i), o.boldFirst && i === 0, ri < rows.length - 1)) }))] });
}
const SP = (n = 60) => new Paragraph({ spacing: { before: 0, after: n }, children: [] });

// =============================================================== contenu
const C = [];

// ---- titre + résumé
C.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 30 }, children: [
  new TextRun({ text: 'ATHÉNA v2 : risque de blessure et fatigue sportive', font: FONT, size: 36, bold: true, color: BLUE })] }));
C.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 40 }, children: [
  new TextRun({ text: 'Ce que la profondeur des arbres révèle sur le plafond de prédictibilité d’un jeu synthétique', font: FONT, size: 24, italics: true, color: '26343F' })] }));
C.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 70 }, children: runs(
  '**ARiA Tech** (Dakar) · David Junior Adejo, Momodou Chérif Diallo · PARC 2026 AI League · Rapport final · Octobre 2026', { size: 20 }) }));
C.push(P('**Résumé.** ATHÉNA est un prototype d’aide à l’analyse de la charge d’entraînement, du risque de blessure et de la fatigue. Nous étudions deux tâches sur un jeu synthétique dont la structure est connue (n = 6 000, 10 variables, prévalence ≈ 20 %). Une première expérience (E1) compare quatre modèles (régression logistique, forêt aléatoire, deux Gradient Boosting) et ne montre pas d’avantage net du Gradient Boosting (GB) sur la régression logistique : l’IC95 de la différence d’AUC contient 0. Une seconde expérience (E2), dont les hypothèses ont été pré-enregistrées, ne fait varier qu’une variable, la profondeur //d// des arbres, et compare les modèles à un oracle, l’AUC du risque latent sans bruit. Le GB de profondeur 1 est le meilleur (AUC ' + fr(p2.gb_fixe['1'].moyenne) + ' ; ' + frs(p2.diff_profondeur1_vs_lr.gb_fixe.moyenne) + ' sur la régression logistique, IC95 ' + ic(p2.diff_profondeur1_vs_lr.gb_fixe.ic95) + '), chaque niveau d’interaction supplémentaire dégrade l’AUC, et l’écart à l’oracle (' + fr(p2.oracle.moyenne) + ') n’est que de ' + fr(p2.ecart_oracle.gb_fixe.moyenne) + ' : le plafond d’environ 0,70 vient du bruit du générateur et non du modèle. Une expérience complémentaire (E3), avec une interaction ajoutée au risque, montre que la profondeur 2 est le choix robuste quand la structure des données est inconnue. Ces résultats portent sur des données synthétiques et ne constituent pas une validation clinique.', { after: 40, run: { size: 20 } }));

// ---- 1
C.push(H1('1. Définition de la tâche, protocole d’évaluation et données'));
C.push(H2('1.1 Tâches'));
C.push(P('À partir de dix variables d’entrée //x// (ACWR, charge chronique, fréquence cardiaque de repos, sommeil, âge, IMC, antécédents de blessure, déficit calorique, lactate, intensité moyenne sur 7 jours), ATHÉNA résout deux tâches : **(A) classification binaire**, prédire //y// = 1 si une blessure survient, avec un score servant à classer les profils par risque ; **(B) régression**, prédire un score de fatigue //f// ∈ [0 ; 10]. Le système est une aide à la discussion pour l’entraîneur, pas un dispositif médical. Le cadre est celui des approches d’apprentissage automatique pour la prédiction du risque de blessure recensées par Leckey et al. [1].'));
C.push(H2('1.2 Données : un générateur synthétique à vérité terrain connue'));
C.push(P('Faute de données prospectives étiquetées accessibles, nous utilisons un générateur (//models/train_injury_model.py//) dont nous contrôlons la structure. Le risque latent est une **somme de termes univariés en « charnière »** : //r//(x) = 2,6·max(ACWR − 1,3 ; 0)^{1,4} + 1,1·max(0,8 − ACWR ; 0) + 0,9·max(7 − sommeil ; 0) + 0,55·antécédents + … ; la blessure est tirée par //y// ~ Bernoulli(σ(//r// + ε − 3)), avec ε ~ N(0 ; 0,9²) non observé. La fatigue est //f// = écrêtage de [3 + 2,2·max(ACWR − 1,1 ; 0) + 1,4·max(7,2 − sommeil ; 0) + … + ε′] dans [0 ; 10], avec ε′ ~ N(0 ; 0,8²). Deux variables (charge chronique, FC de repos) n’ont aucun effet : ce sont des distracteurs. Les formes en U de l’ACWR (sous-charge et surcharge) et la dette de sommeil s’inspirent de la littérature sur la charge [6] ; les coefficients sont choisis par nous et ne sont pas estimés sur des athlètes.'));
C.push(P('**Avantage méthodologique : un plafond mesurable.** Comme //P//(y = 1 | x) = E[σ(//r// + ε − 3)] est croissante en //r//(x), classer par //r//(x) sans bruit maximise l’AUC en population (sur un échantillon fini, un modèle peut le dépasser par hasard). Cet **oracle** donne le meilleur AUC atteignable : tout écart entre un modèle et l’oracle est de l’erreur d’apprentissage, le reste est du bruit irréductible. Il n’existe que parce que les données sont synthétiques ; nous ne le transposons pas au monde réel.'));
C.push(H2('1.3 Protocole d’évaluation'));
C.push(P('La métrique principale est le **ROC-AUC** (indépendante du seuil et de la prévalence) ; nous rapportons aussi précision, rappel et F1 (classification) et MAE, RMSE, R² (fatigue). Les modèles sont entraînés avec une pondération équilibrée des classes. Les écarts-types sont calculés sur les répétitions ; les comparaisons de modèles s’appuient sur des **différences appariées** avec intervalles de confiance à 95 % (IC95).'));
C.push(IMG('fig1_tache_protocole.png', 4.4));
C.push(CAP('**Figure 1.** Tâche et protocole. Partition 80/20 (4 800 / 1 200, graine 42) pour le test principal, puis répétitions (validation croisée à 5 plis, 10 partitions, 10 jeux générés indépendamment), jeu inédit (graine 123, n = 3 000), robustesse et plafond oracle.'));

// ---- 2
C.push(H1('2. Modèle'));
C.push(H2('2.1 Gradient Boosting et fonction de perte'));
C.push(P('Le modèle de production est un Gradient Boosting [2] : une somme d’arbres de faible profondeur ajustés séquentiellement sur le gradient négatif de la perte (Figure 2, haut) :'));
C.push(EQ('F_{M}(x) = F_{0} + ν Σ_{m=1..M} h_{m}(x),     p(x) = σ(F_{M}(x))'));
C.push(P('où //F//_{0} est la log-cote de base, //h//_{m} un arbre de profondeur //d//, ν le pas d’apprentissage et //M// le nombre d’arbres. **Perte (classification)** : log-loss pondérée, //L// = −Σ //w//_{i} [//y//_{i} log //p//_{i} + (1 − //y//_{i}) log(1 − //p//_{i})] / Σ //w//_{i}, avec des poids « équilibrés » //w//_{i} = //n// / (2 //n//_{c(i)}). Chaque arbre ajuste donc //w//_{i}(//y//_{i} − //p//_{i}). **Perte (fatigue)** : erreur quadratique. Les références de comparaison sont la régression logistique (même perte, //F// linéaire en variables standardisées), la forêt aléatoire [3] et HistGradientBoosting, tous issus de scikit-learn [4].'));
C.push(IMG('fig2_modele.png', 5.4));
C.push(CAP('**Figure 2.** Haut : structure du modèle et réglages étudiés. Bas : contributions de l’ACWR et du sommeil au risque, pour le générateur (noir), la meilleure droite (modèle linéaire, gris) et un GB de profondeur 1 appris (bleu). Une souche apprend la forme en charnière sans l’avoir supposée ; une droite ne le peut pas.'));
C.push(TABLE(['Modèle', 'Ce qu’il peut représenter', 'Rôle dans l’étude'], [
  ['Régression logistique / linéaire', 'Effets linéaires et additifs des variables brutes', 'Baseline de E1 et E2'],
  ['Random Forest [3]', 'Interactions, moyenne d’arbres profonds indépendants', 'Comparaison dans E1'],
  ['HistGradientBoosting', 'Interactions ; gère nativement les valeurs manquantes', 'Comparaison dans E1'],
  ['Gradient Boosting [2], //d// = 1', 'Fonction additive non linéaire (escaliers par variable)', 'Modèle étudié dans E2'],
  ['Gradient Boosting [2], //d// ≥ 2', 'Interactions d’ordre ≤ //d//', 'Conditions de E2 ; //d// = 3 dans E1'],
], [2800, 4300, 2260], { boldFirst: true, left: [1, 2] }));
C.push(CAP('**Tableau 1.** Modèles comparés et ce que chacun peut représenter.'));
C.push(H2('2.2 Les parties que nous avons modifiées ou étudiées'));
C.push(P('**(i) La profondeur //d// comme ordre d’interaction.** Un arbre de profondeur //d// ne peut combiner que //d// variables : //d// = 1 donne un modèle **additif** //F//(x) = //F//_{0} + Σ_{j} //g//_{j}(x_{j}), où chaque //g//_{j} est une fonction en escalier apprise (Figure 2, bas). C’est le levier central de l’expérience E2. **(ii) Pondération équilibrée** : elle revient à apprendre à prévalence 0,5 ; pour retrouver une probabilité, nous appliquons une **correction de prévalence** (règle de Bayes) vers la prévalence attendue (≈ 20 %), après une **calibration isotonique** à 5 plis [5] (qui, ici, ne change presque pas le score de Brier : 0,2204 → 0,2203 ; le gain vient de la correction de prévalence). **(iii) Entrées robustes** : validation des domaines, écrêtage et imputation par le milieu du domaine d’entraînement ; cela compte quand des mesures manquent (section 4.3). **(iv) Hyperparamètres de production**, choisis par RandomizedSearchCV (5 plis, 25 tirages, entraînement sur 6 400 profils) : risque de blessure, au plus 330 arbres (arrêt précoce actif), ν = 0,0246, //d// = 2, 36 échantillons minimum par feuille, sous-échantillonnage 0,84 ; fatigue, au plus 164 arbres, ν = 0,0697, //d// = 2. L’espace de recherche était //d// ∈ {2, 3, 4} : //d// = 1 n’a donc jamais été essayé en production. Dans E1 et E2, le GB est en revanche une **configuration de référence fixe** (180 arbres, ν = 0,08), distincte de celle de production, afin de ne faire varier qu’un facteur à la fois.'));
C.push(P('Le démonstrateur de pose 2D (MediaPipe) et l’interface Streamlit complètent le prototype ; ils ne sont pas évalués ici et ne prétendent à aucune valeur clinique.'));

// ---- 3
C.push(H1('3. Expérience'));
C.push(H2('3.1 Point de départ : E1 n’était pas concluante'));
C.push(P('E1 testait H1 (« les ensembles d’arbres surpassent la régression logistique en AUC »). Le bootstrap apparié donne +0,019 d’AUC pour le GB, mais avec un IC95 de ' + ic(e1.test_hypothese.ic95_diff_auc) + ' qui contient 0 : H1 n’est pas confirmée. Plutôt que d’en rester à « pas de différence », nous avons cherché //pourquoi// à l’aide d’une propriété connue du jeu (additivité du risque) et d’une propriété du modèle (la profondeur borne l’ordre d’interaction).'));
C.push(H2('3.2 Question de recherche et hypothèses (E2)'));
C.push(P('**Question (RQ).** Comment la profondeur maximale //d// des arbres, c’est-à-dire l’ordre d’interaction qu’un Gradient Boosting peut représenter, module-t-elle son AUC en généralisation, et quelle part de l’écart à l’AUC maximale possible est due au modèle plutôt qu’au bruit des données ?'));
C.push(P('Les hypothèses ont été fixées et datées (//docs/E2_preenregistrement.md//) **avant la première exécution**, avec leurs règles de décision (fichier horodaté, non certifié par un tiers). Le risque du générateur est additif (aucun terme croisé) et le bruit ε est non observé :'));
C.push(TABLE(['Hyp.', 'Propriété mobilisée', 'Prédiction', 'Réfutée si'], [
  ['H2', 'Risque additif (aucun terme croisé) ; un arbre de profondeur //d// code des interactions d’ordre ≤ //d//', 'Pour //d// ∈ {2, 3, 4, 6}, borne haute de l’IC95 de ΔAUC(//d// − 1) < +0,010', 'Un //d// > 1 gagne ≥ 0,010 avec une borne basse > 0'],
  ['H3', 'Bruit ε non observé : l’oracle est le meilleur AUC atteignable', 'Écart moyen oracle − meilleure profondeur < 0,030', 'Écart ≥ 0,030'],
  ['H4', 'Charnières : la régression logistique est linéaire en les variables brutes ; une souche apprend des marches', 'Borne basse de l’IC95 de ΔAUC(//d// = 1 − régression logistique) > 0', 'Borne haute < 0 ; non concluante si l’IC contient 0'],
], [650, 3300, 2860, 2550], { boldFirst: true, left: [1, 2, 3] }));
C.push(CAP('**Tableau 2.** Hypothèses falsifiables de E2 et règles de décision, fixées avant l’exécution. Seuils choisis avant l’exécution : 0,010 d’AUC est l’écart que nous jugeons négligeable en pratique ; 0,030 est la marge au-delà de laquelle l’écart à l’oracle serait attribué au modèle.'));
C.push(H2('3.3 Variables, conditions et baselines'));
C.push(TABLE(['Rôle', 'Définition'], [
  ['Variable indépendante', 'Profondeur maximale des arbres //d// ∈ {1, 2, 3, 4, 6} (5 conditions)'],
  ['Variables contrôlées', 'ν = 0,08 ; 180 arbres ; pondération équilibrée ; mêmes 10 variables, mêmes données et mêmes partitions pour toutes les conditions'],
  ['Variante E2b', 'Arrêt précoce (≤ 600 arbres, patience 10, validation 15 %) : chaque profondeur reçoit un budget d’arbres adapté'],
  ['Variable dépendante', 'ROC-AUC sur le jeu de test ; en complément, nombre d’arbres retenus'],
  ['Baselines', 'Régression logistique (référence de E1) ; **oracle** (AUC du risque latent sans bruit) ; GB //d// = 3 de E1'],
], [2300, 7060], { boldFirst: true, left: [1] }));
C.push(SP(50));
C.push(H2('3.4 Méthodologie et code'));
C.push(P('**Protocole primaire (P2)** : 10 jeux générés indépendamment (graines 1000 à 1009, n = 6 000), chacun partitionné 80/20 ; IC95 de Student (9 degrés de liberté) sur les différences appariées entre jeux. L’appariement est nécessaire car l’AUC de l’oracle varie fortement d’un jeu à l’autre (' + fr(p2.oracle.min) + ' à ' + fr(p2.oracle.max) + '). **Protocole de cohérence (P1)** : jeu de E1 et ses 10 partitions ; elles se recouvrent, donc P1 ne sert pas à décider. Un garde-fou (//verifier_generateur//) vérifie que le risque latent recalculé reproduit exactement les étiquettes du générateur. Code : Python 3.12, scikit-learn 1.9.1 ; //python -m models.experiment_profondeur// (≈ 2,5 min sur 2 cœurs) écrit //models/artifacts/experiment_profondeur.json// ; deux exécutions successives donnent des résultats identiques.', { align: AlignmentType.LEFT }));

C.push(H2('3.5 Menaces à la validité de l’expérience'));
C.push(P('**Fuite d’information** : les hyperparamètres de E2 sont fixés a priori, aucun n’est ajusté sur le jeu de test ; l’arrêt précoce utilise une validation interne prise dans l’apprentissage. **Comparaisons multiples** : les hypothèses (E2 et E3) et leurs règles sont fixées à l’avance ; H2 exige que les quatre contrastes soient simultanément conformes, ce qui est plus exigeant qu’un contraste isolé. **Puissance** : dix jeux donnent des IC de largeur ≈ 0,01 à 0,04, suffisants pour trancher des écarts de l’ordre de 0,01 à 0,07 mais pas des écarts plus fins. **Validité externe** : le générateur est additif par construction ; E2 teste donc une relation entre la structure du modèle et celle des données, pas une propriété du sport (voir 4.5 et 4.6).', { after: 70 }));
// ---- 4
C.push(H1('4. Résultats et discussion'));
C.push(H2('4.1 E1 : familles de modèles'));
const cl = e1.classification_injury_risk, ms = e1.variance_multi_seed.classification_roc_auc;
const nm = ['Régression logistique', 'Random Forest', 'Gradient Boosting (d = 3)', 'HistGradientBoosting'];
const keys = Object.keys(ms);
const cible = ['LR', 'RF', 'GB', 'HGB'];
C.push(TABLE(['Modèle', 'AUC (test)', 'F1', 'Rappel', 'AUC moy. ± σ (10 partitions)'],
  cl.map((r, i) => [nm[i], fr(r.roc_auc, 4), fr(r.f1, 4), fr(r.recall, 4), fr(ms[keys[i]].moyenne) + ' ± ' + fr(ms[keys[i]].ecart_type)])
    .concat([['GB //d// = 1 (E2, jeu de E1)', '—', '—', '—', fr(p1.gb_fixe['1'].moyenne) + ' ± ' + fr(p1.gb_fixe['1'].ecart_type)]]),
  [2900, 1300, 1000, 1100, 3060]));
C.push(CAP('**Tableau 3.** E1, classification (test principal, n = 1 200, et moyenne de 10 partitions du même jeu). Les écarts entre modèles (jusqu’à 0,024 pour RF − régression logistique sur le test principal) sont du même ordre que la dispersion σ (≈ 0,02).'));
C.push(P('En classification, aucun modèle ne se détache : Random Forest est premier sur le test principal (0,695), devant le GB d’étude (0,690), HistGB (0,686) et la régression logistique (0,671). Pour la **fatigue**, le GB atteint MAE = ' + fr(e1.regression_fatigue[2].mae, 4) + ' et R² = ' + fr(e1.regression_fatigue[2].r2, 3) + ', contre ' + fr(e1.regression_fatigue[0].mae, 4) + ' et ' + fr(e1.regression_fatigue[0].r2, 3) + ' pour la régression linéaire ; le plancher de bruit est ' + fr(e2.plancher_bruit_fatigue.mae, 4) + ' (R² ' + fr(e2.plancher_bruit_fatigue.r2, 3) + '). **Le GB est donc à 0,005 de MAE du minimum théorique** : la fatigue est essentiellement résolue sur ce jeu, et l’avantage sur le modèle linéaire (−0,13 de MAE) vient des seuils que la droite ne représente pas.'));

C.push(IMG('fig3_resultats_e1.png', 5.5));
C.push(CAP('**Figure 3.** (a) AUC moyen des modèles de E1 et du GB de profondeur 1 (E2) ; le trait noir est l’oracle (plafond, 0,709 sur les partitions du jeu de E1 ; 0,708 sur les 10 jeux indépendants de E2). (b) MAE de la fatigue ; le trait noir est le plancher de bruit, MAE du prédicteur qui connaît la partie déterministe du générateur.'));
C.push(H2('4.2 E2 : l’effet de la profondeur'));
const pr = e2.protocole.profondeurs;
C.push(TABLE(['//d//', 'AUC, 180 arbres (moy. ± σ)', 'ΔAUC vs //d// = 1 [IC95]', 'AUC, arrêt précoce', 'Arbres retenus'],
  pr.map(d => [String(d), fr(p2.gb_fixe[d].moyenne) + ' ± ' + fr(p2.gb_fixe[d].ecart_type),
    d === 1 ? 'référence' : frs(p2.diff_vs_profondeur1.gb_fixe[d].moyenne) + ' ' + ic(p2.diff_vs_profondeur1.gb_fixe[d].ic95),
    fr(p2.gb_arret[d].moyenne), String(Math.round(p2.n_arbres_arret[d]))])
    .concat([['Rég. logistique', fr(p2.lr.moyenne) + ' ± ' + fr(p2.lr.ecart_type), 'GB //d// = 1 : ' + frs(p2.diff_profondeur1_vs_lr.gb_fixe.moyenne) + ' ' + ic(p2.diff_profondeur1_vs_lr.gb_fixe.ic95), '—', '—'],
      ['Oracle', fr(p2.oracle.moyenne) + ' ± ' + fr(p2.oracle.ecart_type), 'GB //d// = 1 : −' + fr(p2.ecart_oracle.gb_fixe.moyenne) + ' ' + ic([-p2.ecart_oracle.gb_fixe.ic95[1], -p2.ecart_oracle.gb_fixe.ic95[0]]), '—', '—']]),
  [1400, 2200, 3260, 1400, 1100], { shade: t => false }));
C.push(CAP('**Tableau 4.** E2, protocole P2 (10 jeux indépendants). Les IC95 sont des IC de Student sur différences appariées.'));
C.push(P('**H2 est confirmée** : les bornes hautes sont ' + [2, 3, 4, 6].map(d => fr(p2.diff_vs_profondeur1.gb_fixe[d].ic95[1])).join(', ') + ' pour //d// = 2, 3, 4, 6, toutes très inférieures à +0,010 ; l’AUC décroît de façon monotone avec la profondeur (' + fr(p2.gb_fixe['1'].moyenne) + ' → ' + fr(p2.gb_fixe['6'].moyenne) + '). L’arrêt précoce atténue la perte (//d// = 6 : ' + fr(p2.gb_arret['6'].moyenne) + ' au lieu de ' + fr(p2.gb_fixe['6'].moyenne) + ') mais ne l’efface pas, et //d// = 1 reste le meilleur : la dégradation n’est pas un simple artefact du nombre d’arbres. **H3 est confirmée** : la meilleure condition est à ' + fr(p2.ecart_oracle.gb_fixe.moyenne) + ' de l’oracle (IC95 ' + ic(p2.ecart_oracle.gb_fixe.ic95) + '), bien sous le seuil de 0,030 ; la régression logistique en est à ' + fr(p2.ecart_oracle.lr.moyenne) + '. **H4 est confirmée** : //d// = 1 gagne ' + fr(p2.diff_profondeur1_vs_lr.gb_fixe.moyenne) + ' d’AUC sur la régression logistique (IC95 ' + ic(p2.diff_profondeur1_vs_lr.gb_fixe.ic95) + ') ; sur le jeu de E1 (P1), l’écart est de ' + fr(p1.diff_profondeur1_vs_lr.gb_fixe.moyenne) + ' ' + ic(p1.diff_profondeur1_vs_lr.gb_fixe.ic95.map(Number)) + '.'));
C.push(P('**Cela explique E1.** Le GB de E1 utilisait //d// = 3 : à cette profondeur, il ne se distingue plus de la régression logistique (ΔAUC = ' + frs(p2.diff_profondeur3_vs_lr.moyenne) + ', IC95 ' + ic(p2.diff_profondeur3_vs_lr.ic95) + '). Il perdait l’avantage de la non-linéarité en payant la variance d’interactions inexistantes. Le « résultat nul » de E1 était donc un défaut de réglage et non l’absence de non-linéarité à exploiter.'));

C.push(IMG('fig4_profondeur.png', 5.5));
C.push(CAP('**Figure 4.** (a) AUC selon la profondeur, avec l’oracle et la régression logistique. (b) ΔAUC par rapport à //d// = 1 ; le trait rouge est le seuil de décision de H2 (+0,010) : aucun intervalle ne s’en approche.'));
C.push(H2('4.3 Apprentissage, bruit et données manquantes'));
C.push(P('La courbe d’apprentissage (GB d’étude, //d// = 3) plafonne à ≈ 0,676 dès 3 600 exemples : ce plateau reste sous le GB additif (' + fr(p2.gb_fixe['1'].moyenne) + ') et sous l’oracle (' + fr(p2.oracle.moyenne) + '), donc ajouter des données ne comble pas l’écart ; c’est la structure du modèle, pas le volume, qui limite. À 20 % de valeurs manquantes, l’AUC du modèle de production passe de ' + fr(av.robustesse.manquants[0].production_imputation_milieu) + ' à ' + fr(av.robustesse.manquants[2].production_imputation_milieu) + ', alors que le GB d’étude sans imputation descend à ' + fr(av.robustesse.manquants[2].gb_imputation_milieu) + ' : la **chaîne de pré-traitement compte autant que l’algorithme**. Un bruit de mesure d’écart-type 0,5 (en unités standardisées) coûte ≈ 0,04 d’AUC. Sur le jeu inédit de graine 123 (n = 3 000), le modèle de production atteint ' + fr(av.test_donnees_inedites['Gradient Boosting (production)'].roc_auc) + '. Les niveaux absolus d’AUC de production diffèrent selon le jeu de test (0,666 sur la réserve interne de l’entraînement, 0,709 sur le jeu des courbes de robustesse, 0,684 sur le jeu inédit) : seules les comparaisons au sein d’une même courbe sont interprétées.'));
C.push(IMG('fig5_apprentissage_robustesse.png', 5.9));
C.push(CAP('**Figure 5.** (a) Courbe d’apprentissage (GB d’étude, //d// = 3) : plateau vers 3 600 exemples, sous l’oracle. (b) AUC avec valeurs manquantes : modèle de production avec imputation par le milieu du domaine, HistGB natif et GB d’étude sans cette imputation. (c) Bruit gaussien ajouté aux entrées.'));
C.push(H2('4.4 Calibration et importance des variables'));
const cal = av.calibration_production;
C.push(P('Le score brut n’est pas une probabilité : sa moyenne est ' + fr(cal.score_moyen_production) + ' pour une fréquence observée de ' + fr(cal.frequence_reelle_blessure) + '. La correction de prévalence ramène le score de Brier de ' + fr(cal.brier_score_brut) + ' à ' + fr(cal.brier_apres_correction_prevalence, 4) + ', sous la référence constante (' + fr(cal.brier_reference_constante, 4) + '). Le classement des variables par le modèle est fidèle à la vérité du générateur (sommeil, antécédents, ACWR en tête ; charge chronique et âge ≈ 0), avec une fausse importance résiduelle de la FC de repos (0,007 contre 0 pour l’oracle). Cette concordance valide la lecture « explicable » du modèle **sur ce jeu** ; elle ne prouve aucune causalité.'));

C.push(IMG('fig6_calibration_importance.png', 5.9));
C.push(CAP('**Figure 6.** (a) Fiabilité des probabilités du modèle de production avant et après correction de prévalence. (b) Importance par permutation du GB de référence de E1 (//d// = 3) comparée à celle de l’oracle ; ρ de Spearman = ' + fr(e2.importance_vs_verite.spearman_rho, 2) + ' (p = ' + fr(e2.importance_vs_verite.p_value, 3) + ').'));
C.push(H2('4.5 E3 : et si les données contiennent une interaction ?'));
const d21 = c => R3[c].diff.gb_fixe['2_vs_1'], d32 = c => R3[c].diff.gb_fixe['3_vs_2'];
C.push(P('Un jeu inconnu peut être plus complexe que le nôtre. Nous avons donc ajouté au risque un **terme croisé** //c//·z_{ACWR}·z_{sommeil} (interaction d’ordre exactement 2, d’écart-type //c//) pour //c// ∈ {0 ; 0,25 ; 0,5 ; 1,0}, avec la même méthode que E2 (10 jeux indépendants par //c//, graines 2000 à 2009, prévalence recalée à 20 %) et des hypothèses pré-enregistrées (//docs/E3_preenregistrement.md//). **H8 (réplication de E2) est confirmée** : à //c// = 0, //d// = 2 perd ' + fr(-d21('0.0').moyenne) + ' d’AUC face à //d// = 1 (IC95 ' + ic(d21('0.0').ic95) + '). **H5 est non concluante** : à //c// = 0,5 le gain de //d// = 2 est de ' + frs(d21('0.5').moyenne) + ' (IC95 ' + ic(d21('0.5').ic95) + ') et ne passe pas le seuil, alors qu’à //c// = 1,0 il atteint ' + frs(d21('1.0').moyenne) + ' (IC95 ' + ic(d21('1.0').ic95) + ') ; une interaction modérée est donc invisible à cette taille d’échantillon. **H6 est confirmée** : //d// = 3 n’apporte rien de plus que //d// = 2 (bornes hautes ≤ ' + fr(Math.max(...Object.values(V3.H6.bornes_hautes_3_vs_2))) + ', seuil 0,010). **H7 est confirmée** : le regret maximal de //d// = 2 est ' + fr(V3.H7.regret_max_d2, 4) + ' (seuil 0,012), contre ' + fr(V3.H7.regret_max_d1, 4) + ' pour //d// = 1.'));
C.push(IMG('fig7_interaction.png', 5.5));
C.push(CAP('**Figure 7.** E3. (a) AUC selon la profondeur pour quatre forces d’interaction //c// ; les pointillés sont l’oracle de chaque condition. (b) Regret = AUC du meilleur //d// − AUC de //d// ; trait rouge : seuil de H7 (0,012).'));
C.push(P('**Lecture.** La profondeur 1, optimale sur un risque additif, perd ' + fr(V3.H7.regret_max_d1, 3) + ' d’AUC si l’interaction est forte ; la profondeur 2 ne perd jamais plus de ' + fr(V3.H7.regret_max_d2, 3) + ' : c’est le choix qui minimise le pire regret (critère minimax). C’est aussi la profondeur retenue par la recherche d’hyperparamètres de production (section 2.2).'));
C.push(H2('4.6 Discussion et limites'));
C.push(P('**Ce que l’on retient.** (1) L’AUC d’environ 0,70 n’est pas une faiblesse du modèle : l’oracle plafonne à ' + fr(p2.oracle.moyenne) + ' à cause du bruit non observé, et le GB additif l’approche à ' + fr(p2.ecart_oracle.gb_fixe.moyenne) + '. (2) Choisir un modèle plus complexe a un coût mesurable quand la structure des données est plus simple que celle du modèle : chaque niveau d’interaction retire de l’AUC. (3) Le réglage (profondeur) et le pré-traitement pèsent plus que le choix de la famille d’algorithmes. La profondeur 2 retenue en production est un choix robuste (E3) : elle perd au plus ' + fr(V3.H7.regret_max_d2, 3) + ' d’AUC si le risque est additif et gagne jusqu’à ' + fr(d21('1.0').moyenne, 3) + ' s’il contient une interaction forte.'));
C.push(P('**Limites.** (i) La conclusion « profondeur 1 est optimale » est vraie //par construction du générateur// (risque additif) : elle illustre un principe, adapter la structure du modèle à celle des données et la comparer à un plafond, et non une loi du sport ; sur des données réelles, des interactions (par exemple ACWR × sommeil) peuvent exister. E3 teste cette lecture avec une interaction simulée (4.5) ; une interaction réelle reste à démontrer sur des données de terrain. (ii) Aucune donnée prospective réelle n’est utilisée : aucune performance clinique n’est revendiquée. (iii) Dix jeux donnent des IC larges (écart-type ≈ 0,017 par jeu). (iv) Dans l’analyse d’erreurs du projet, environ la moitié des blessures simulées sont manquées (122 faux négatifs pour 129 vrais positifs) : un score faible n’est pas une garantie d’absence de risque. (v) Les mesures de pose 2D dépendent du cadrage.'));
C.push(P('**Conclusion.** Sur un jeu synthétique à vérité connue, ATHÉNA prédit la fatigue presque au minimum théorique et le risque de blessure presque au plafond imposé par le bruit ; le gain vient d’un réglage qui respecte la structure additive des données (//d// = 1) ; si les données contiennent une interaction, la profondeur 2 de production est le choix le plus robuste (E3). Six des sept hypothèses pré-enregistrées sont confirmées et une est non concluante (H5). La prochaine étape est la validation sur des données réelles consenties.'));

// ---- 5 références (nouvelle page)
C.push(new Paragraph({ children: [new PageBreak()] }));
C.push(H1('5. Références'));
const REFS = [
  '[1] Leckey C, van Dyk N, Doherty C, Lawlor A, Delahunt E. Machine learning approaches to injury risk prediction in sport: a scoping review with evidence synthesis. //Br J Sports Med//. 2025;59(7):491–500. doi:10.1136/bjsports-2024-108576.',
  '[2] Friedman JH. Greedy function approximation: a gradient boosting machine. //Annals of Statistics//. 2001;29(5):1189–1232.',
  '[3] Breiman L. Random forests. //Machine Learning//. 2001;45(1):5–32. doi:10.1023/A:1010933404324.',
  '[4] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. //Journal of Machine Learning Research//. 2011;12:2825–2830.',
  '[5] Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. //Proceedings of the 22nd International Conference on Machine Learning (ICML)//. 2005:625–632.',
  '[6] Gabbett TJ. The training–injury prevention paradox: should athletes be training smarter and harder? //Br J Sports Med//. 2016;50(5):273–280.',
  '[7] PARC 2026 AI League. Sports Insights Enabled by AI/ML : rapport final, critères et livrables. https://parc-ai-league.github.io/PARC2026-AI-League/ (consulté le 10 octobre 2026).',
  '[8] ARiA Tech. ATHÉNA v2 : code source, artefacts et scripts d’expériences (archive athena_v2_PARC2026_v10.zip).',
];
REFS.forEach(r => C.push(P(r, { align: AlignmentType.LEFT, after: 60, indent: { left: 420, hanging: 420 } })));
C.push(H2('Annexe : traçabilité et reproduction'));
C.push(P('Tous les chiffres des tableaux et figures sont lus dans des fichiers JSON régénérables : //experiment_results.json// (E1), //experiment_profondeur.json// (E2), //experiment_interaction.json// (E3), //advanced_analysis.json// (robustesse, calibration, courbe d’apprentissage) et //metadata.json// (modèle de production). Commandes, depuis la racine du projet : //python -m models.experiment_comparaison// (E1) ; //python -m models.experiment_profondeur// (E2) ; //python -m models.experiment_interaction// (E3, ≈ 5 min) ; //python -m reports.figures_rapport// (figures). Hypothèses et seuils de E2 et E3 : //docs/E2_preenregistrement.md// et //docs/E3_preenregistrement.md//, écrits avant les premières exécutions ; il consigne aussi une prédiction personnelle erronée (nous attendions un écart à l’oracle voisin de 0,03 ; il est de 0,007). Les artefacts de modèles dépendent de la version de scikit-learn.', { align: AlignmentType.LEFT, after: 60 }));
C.push(P('**Cadre éthique.** ATHÉNA est un prototype d’aide à la décision, sans prétention diagnostique. Toute collecte future auprès de sportifs exigera consentement éclairé, minimisation des données personnelles, stockage sécurisé et validation par des professionnels de santé et du sport.', { align: AlignmentType.LEFT }));

const doc = new Document({
  creator: 'ARiA Tech', title: 'ATHÉNA v2 — Rapport final PARC 2026',
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: 'ATHÉNA v2 · ARiA Tech · PARC 2026 AI League · page ', font: FONT, size: 17, color: '6B7B88' }),
      new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 17, color: '6B7B88' })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log('écrit', OUT, b.length); });
