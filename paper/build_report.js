// Builds paper/report_draft.docx from the text below and the figures in figures/.
//   node paper/build_report.js        (needs the npm package "docx")
// Inline math is written between $...$: letters are set in italics, _x / _{xy} are
// subscripts, ^x / ^{xy} superscripts. **text** is bold.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, ShadingType, Footer, PageNumber, TabStopType, LineRuleType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FONT = "Times New Roman";
const BODY = 20;                 // half-points: 10 pt, as in the NeurIPS style
const TEXT_DXA = 7920;           // 5.5 in text width
let figNo = 0, tabNo = 0, eqNo = 0;

// ------------------------------------------------------------------ inline markup
function mathRuns(src, size) {
  const runs = [];
  let i = 0;
  const group = () => {
    if (src[i] === "{") { const j = src.indexOf("}", i); const g = src.slice(i + 1, j); i = j + 1; return g; }
    return src[i++];
  };
  while (i < src.length) {
    const c = src[i];
    if (c === "_" || c === "^") {
      i++;
      const g = group().replace(/[_^]/g, "");
      const roman = g.length > 1 && /^[A-Za-z ,]+$/.test(g);
      for (const ch of g) runs.push(new TextRun({ text: ch, font: FONT, size, italics: !roman && /[A-Za-zλμφ]/.test(ch),
        subScript: c === "_", superScript: c === "^" }));
    } else {
      i++;
      let t = c;
      while (i < src.length && /[\u0300-\u036f]/.test(src[i])) t += src[i++];   // keep combining accents
      runs.push(new TextRun({ text: t, font: FONT, size, italics: /[A-Za-zλμφσ]/.test(c) }));
    }
  }
  return runs;
}

function runs(text, size = BODY, base = {}) {
  const out = [];
  const parts = text.split(/(\$[^$]+\$|\*\*[^*]+\*\*)/);
  for (const p of parts) {
    if (!p) continue;
    if (p.startsWith("$")) out.push(...mathRuns(p.slice(1, -1), size));
    else if (p.startsWith("**")) out.push(new TextRun({ text: p.slice(2, -2), font: FONT, size, bold: true, ...base }));
    else out.push(new TextRun({ text: p, font: FONT, size, ...base }));
  }
  return out;
}

// ------------------------------------------------------------------ blocks
const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.size || BODY), alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 100, line: 250, lineRule: LineRuleType.AUTO }, indent: opts.indent,
  keepNext: opts.keepNext,
});
const H1 = (text) => new Paragraph({ children: [new TextRun({ text, font: FONT, size: 24, bold: true })],
  spacing: { before: 200, after: 100 }, keepNext: true });
const H2 = (text) => new Paragraph({ children: [new TextRun({ text, font: FONT, size: 20, bold: true })],
  spacing: { before: 140, after: 60 }, keepNext: true });
const Eq = (text) => {
  eqNo += 1;
  return new Paragraph({
    children: [new TextRun({ text: "\t", font: FONT }), ...mathRuns(text, BODY),
      new TextRun({ text: `\t(${eqNo})`, font: FONT, size: BODY })],
    tabStops: [{ type: TabStopType.CENTER, position: TEXT_DXA / 2 }, { type: TabStopType.RIGHT, position: TEXT_DXA }],
    spacing: { before: 60, after: 100 },
  });
};
const Code = (lines) => lines.map((l, k) => new Paragraph({
  children: [new TextRun({ text: l || " ", font: "Courier New", size: 16 })],
  shading: { type: ShadingType.CLEAR, color: "auto", fill: "F3F2EE" },
  spacing: { before: k === 0 ? 60 : 0, after: k === lines.length - 1 ? 120 : 0 },
  indent: { left: 200, right: 200 },
}));

function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function Fig(name, caption, widthIn = 5.5) {
  figNo += 1;
  const file = path.join(ROOT, "figures", `${name}.png`);
  const [w, h] = pngSize(file);
  const width = Math.round(widthIn * 96);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width, height: Math.round(width * h / w) } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
      children: [new TextRun({ text: `Figure ${figNo}. `, font: FONT, size: 18, bold: true }), ...runs(caption, 18)] }),
  ];
}

const rule = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function Tab(caption, header, rows, widths, opts = {}) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, r, c, isHead) => new TableCell({
    width: { size: widths[c], type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 70, right: 70 },
    borders: { left: none, right: none, top: isHead ? rule : none,
      bottom: isHead ? thin : (r === rows.length - 1 ? rule : none) },
    children: [new Paragraph({ alignment: c === 0 ? AlignmentType.LEFT : (opts.align || AlignmentType.CENTER),
      children: runs(text, 17, isHead ? { bold: true } : {}) })],
  });
  return [
    new Paragraph({ keepNext: true, spacing: { before: 140, after: 60 }, alignment: AlignmentType.JUSTIFIED,
      children: [new TextRun({ text: `Table ${tabNo}. `, font: FONT, size: 18, bold: true }), ...runs(caption, 18)] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
      rows: [new TableRow({ tableHeader: true, children: header.map((t, c) => cell(t, -1, c, true)) }),
        ...rows.map((row, r) => new TableRow({ cantSplit: true, children: row.map((t, c) => cell(t, r, c, false)) }))] }),
    new Paragraph({ spacing: { after: 80 }, children: [] }),
  ];
}

// ------------------------------------------------------------------ content
const title = [
  new Paragraph({ border: { top: { style: BorderStyle.SINGLE, size: 24, color: "000000", space: 8 } }, spacing: { after: 0 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({
    text: "Battery health on unseen cell kinds from charge and rest-to-rest capacity ratios, calibrated in context",
    font: FONT, size: 34, bold: true })] }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 8 } }, spacing: { after: 200 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({
    text: "Jorgen Rokne Dahle      Junbo Fu      Kasper Andreas Haugaard      Ziv Behar", font: FONT, size: 20, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({
    text: "CS 182/282A project report — draft for peer review, 24 November 2026", font: FONT, size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Abstract", font: FONT, size: 24, bold: true })] }),
];

const abstract = P(
  "Estimating the state of health (SoH) of a lithium-ion cell from its operating data is hard when the cell belongs to a " +
  "kind, a cell design and chemistry, that no model was trained on. We first tested the route of our proposal: a frozen " +
  "time-series foundation model (Chronos-2) that reads 30-minute driving clips with $K$ labelled clips of the new kind in " +
  "context. Under leave-one-kind-out evaluation on five public ageing datasets, no frozen, LoRA-adapted or learned " +
  "in-context variant beat the mean of the $K$ labels. We then recast the task. Each steady charge and each pair of rests " +
  "yields a label-free capacity ratio, measured against the cell's own early life or against a new cell of its kind; at " +
  "every charge, a 15-number vector of such ratios and their conditions is formed from the record up to that moment, and " +
  "the $K$ labelled examples fit a ridge regression on it. At $K$ = 20 the mean absolute error is 1.49 points of SoH (range " +
  "over example draws 1.36–1.82), against 2.45 for the single-charge ratio used in field studies, 3.80 for our " +
  "re-implementation of the fleet protocol of Park et al. (2026), and 3.89 for the mean label. Measurements taken between " +
  "charges provide most of the gain. A meta-trained MLP embedding did not improve on ridge.",
  { indent: { left: 500, right: 500 }, after: 200 });

// Defined first so that figures are numbered in reading order.
const methodFig = Fig("fig1_method", "The method. (a) The cell's record up to the estimate time $t$: current $I(τ)$ (C-rate, positive when charging), voltage $V(τ)$ (scaled to 0–1 between the datasheet limits) and temperature $T(τ)$, sampled every 10 s at times $τ ≤ t$. An estimate is made at the end of every steady charge. (b) Three kinds of event found in the record: steady charges (constant current for at least 10 min), full charges that start after a rest and end in the constant-voltage taper, and pairs of consecutive rests (|$I$| < 0.02C for at least 5 min). (c) Each event gives a label-free capacity ratio. $r_c$: the charge $Q_{[v_1,v_2]}$ passed while the voltage rises from $v_1$ to $v_2$ during this charge, divided by $Q̄^{ref}_{[v_1,v_2]}$, the mean of the same quantity over the cell's earliest 5–10 steady charges. $r_f$: the charge $Q_{fill}(V_0)$ needed to fill the cell from its resting voltage $V_0$, divided by $Q̄^{ref}_{fill}(V_0)$, the same quantity in the cell's earliest full charges started at that voltage. $s$: the charge $ΔQ$ counted between two rests with resting voltages $V_1$ and $V_2$, divided by $ΔQ_{new}(V_1, V_2)$, the charge a new cell of the same kind needs between those voltages, from a curve fitted on the example cells while their health was at least 0.95 (Section 4.2). (d) $x_t$: 15 numbers summarising the latest ratios, their recent history and the conditions of the charge (Appendix B). (e) {($x_k, y_k$)}, $k$ = 1…$K$: the $K$ labelled examples, charges of the new kind's example cells with feature vectors $x_k$ and measured health $y_k$. A ridge regression on them gives the intercept $a$ and weights $w$; the estimate is $ŷ_t = a + w^⊤x_t$, the health as a fraction of the cell's highest measured capacity.");

const intro = [
  H1("1  Introduction"),
  P("The state of health of a battery, its present capacity as a fraction of its capacity when new, sets an electric " +
    "vehicle's range, its resale value and its warranty. California's Advanced Clean Cars II rules require batteries to " +
    "retain 70% of their certified range for eight years and limit how far a vehicle's own health reading may exceed a " +
    "measured value [1]; the EU Battery Regulation requires the state of health to be reported [2]. The readings cars report " +
    "today are unreliable: in an audit of 1,114 vehicles from five manufacturers, the health shown by the battery management " +
    "system correlated with an independent capacity measurement at between ρ = 0.10 and 0.62, and 384 vehicles did not " +
    "report it at all [3]."),
  P("A laboratory measures health by a slow, complete discharge under a reference protocol; a vehicle cannot. It has to " +
    "infer health from the current and voltage it records in use. Two things make this hard. First, every new vehicle " +
    "programme brings a new kind of cell, and models trained on one kind transfer poorly to another: a network trained on one " +
    "battery type erred by 2 to 15 points on four others before adaptation [6]. Second, labels are scarce, since each one " +
    "requires a reference test. We therefore ask: **given $K$ labelled examples from other cells of a new kind and no " +
    "retraining, how accurately can a cell's present health be estimated from its own operating record?**"),
  P("Our proposal answered this with a frozen neural network that reads a 30-minute clip of driving current and voltage " +
    "together with $K$ labelled clips of the same kind, in the in-context-learning setting of Garg et al. [12]. We tested it " +
    "with Chronos-2 [14] as the encoder: frozen embeddings with a ridge head, LoRA adapters [15] of rank 8, 32 and 128, " +
    "Chronos-2's own forecast head as the reader, and a transformer reader trained on disguised episodes so that it must use " +
    "its examples. None beat the mean of the $K$ labels on a held-out kind (Section 4.1, Appendix A). Diagnostics showed why: " +
    "within one kind, a 30-minute clip carries little information about health, so there is little for any learner to " +
    "transfer. We kept the question and changed the input."),
  P("Capacity is visible in data a vehicle already records. During a steady charge, the charge needed to raise the voltage " +
    "across a fixed window is proportional to capacity; between two rests, the resting voltages fix the state of charge at " +
    "both ends and the counted current gives the charge that moved between them. Each such measurement, divided by the same " +
    "quantity for the cell when new, is a label-free ratio that falls almost linearly with health. Kinds differ mainly in the " +
    "scale and offset of that relation, which a few labelled examples can fix. Our contributions are:"),
  P("(i) A negative result with diagnostics: clip-based Chronos-2 estimators, frozen, adapted or reading examples in " +
    "context, do not transfer to unseen kinds in our data.", { indent: { left: 280 }, after: 40 }),
  P("(ii) A causal, label-free feature vector that fuses each charge with what the cell measured since earlier charges, " +
    "calibrated on a new kind by in-context ridge regression: 1.49 points at $K$ = 20 across five unseen kinds.",
    { indent: { left: 280 }, after: 40 }),
  P("(iii) A controlled comparison on identical estimation moments against the single-charge ratio and a re-implementation " +
    "of the protocol of Park et al. [3], and an information ladder that adds one source of information at a time. Against " +
    "the single-charge ratio, the paired gain is 0.96 points (95% bootstrap interval 0.69–1.22).", { indent: { left: 280 }, after: 40 }),
  P("(iv) Code that regenerates every number and figure from the public datasets.", { indent: { left: 280 } }),
  P("Table 1 relates the proposal's hypotheses to what we found."),
  ...Tab("Proposal hypotheses and their status. NASA and the label-shuffle control are not yet run.",
    ["Proposal hypothesis", "Status", "Evidence"],
    [["H1: a frozen model beats the average guess with no examples", "Rejected for clip models", "Chronos-2 ridge on unseen kinds: skill ≤ 0 on every kind (App. A)"],
     ["H2: error falls with $K$ and reaches about 1 point by $K$ = 5", "Partly", "Falls with $K$ (2.25 → 1.49 from $K$ = 5 to 20), not 1 point"],
     ["H3: NASA is the harder test kind", "Not yet tested", "NASA stays locked until the method is frozen"],
     ["H4: examples from another kind raise the error", "In progress", "Label-shuffle and swap controls for the ridge are next"]],
    [2700, 1500, 3720], { align: AlignmentType.LEFT }),
];

const related = [
  H1("2  Background and related work"),
  H2("2.1  Health from charging curves"),
  P("Incremental-capacity and differential-voltage analysis read degradation from the shape of a slow charge curve: peaks " +
    "of $dQ/dV$ shrink and shift as active material and lithium inventory are lost [27, 28]. Machine-learning work builds on " +
    "the same signal. Severson et al. [29] predicted cycle life from the change of the discharge curve between two early " +
    "cycles. Roman et al. [30] estimated health from features of constant-current charging. Wang et al. [4] fitted a " +
    "physics-informed network to features measured just before full charge and reported 0.87% mean absolute percentage " +
    "error on 387 cells from four datasets, including the XJTU cells we use. Che et al. [5] mapped partial charging and " +
    "dynamic discharge to electrode-level states with an encoder–decoder and reported 1.35 points on 236 cells from partial " +
    "charging, and 1.15 points on 92 Stanford driving cells after fine-tuning on about a third of them. All of these train " +
    "and test on the same kind of cell."),
  P("Park et al. [3] brought the charge measurement to the field. From telematics of 1,114 vehicles over 375 days they " +
    "extracted constant-current slow-charging segments (current coefficient of variation below 25%, at least five minutes " +
    "after trimming three at each end, 10–35 °C) and measured the charge passed in a fixed 0.08–0.12 V window per cell, " +
    "normalised by the 90th percentile of the vehicle platform. In a laboratory validation the partial-window charge tracked " +
    "reference capacity with ρ > 0.80. Their limitations section is the starting point of this work: a single charge " +
    "reading varied by 5–7% within one vehicle, so per-vehicle precision was only moderate; seasonal temperature dominated " +
    "month-to-month readings; only 65–93% of vehicles had a qualifying slow charge in a year, and fast charges were " +
    "excluded; and a model fitted on one platform did not transfer to a platform with a different cell design."),
  H2("2.2  Health from driving and from rests"),
  P("Under variable load the voltage moves with the current as well as with the state of charge, so driving data are " +
    "harder to read. Lanubile et al. [7] used energy throughput between charge and discharge to estimate capacity on five " +
    "LG M50T cells (error below 1.6%) with a linear model on five hand-built indicators. Chen et al. [8] used three discharge " +
    "features and a random forest on four NASA random-walk cells (4.6 points). Richardson et al. [9] forecast capacity fade " +
    "on NASA's cells with a Gaussian process (4.3% normalised RMSE). A different classical route avoids the load " +
    "entirely: after a rest the terminal voltage approaches the open-circuit voltage, which fixes the state of charge, so the " +
    "charge counted between two rests divided by the change in state of charge estimates capacity. Plett [20] made this a " +
    "recursive total-least-squares estimator that weighs the uncertainty of both quantities, and the approach appears in " +
    "patents for on-board capacity estimation [31]. It needs an open-circuit-voltage curve of the cell, which is itself " +
    "kind-specific. We use it as one source among several, with the curve taken from new cells of the same kind."),
  H2("2.3  Transfer to new battery kinds"),
  P("Feng et al. [6] are the closest prior work on new kinds. They pre-trained a network on one battery type and adapted it " +
    "at test time on unlabelled charging data of the target type with a physics-guided self-supervised loss. The adapted " +
    "model reached 1.28–1.52 points on three target datasets and 7.95 on a fourth. Sun et al. [11] fine-tuned the TimeGPT " +
    "foundation model on cycling records of 143 batteries. Tan et al. [10] built BatteryLife, a benchmark of 16 datasets and " +
    "990 batteries, for cycle-life prediction across kinds, a different target from present health. Che et al. [5] and " +
    "Park et al. [3] both note that a new cell design needs recalibration. Our setting differs from all of these in what " +
    "the new kind supplies: neither retraining nor unlabelled target data in bulk, but $K$ labelled examples used at test " +
    "time only, together with the record of the cell being estimated."),
  H2("2.4  In-context learning and foundation models"),
  P("Garg et al. [12] showed that a transformer trained on random linear functions infers an unseen function from $K$ " +
    "input–output pairs about as well as least squares. Chronos [13] tokenises a time series and forecasts it with a " +
    "language-model architecture; Chronos-2 [14] adds group attention across related series and covariates, so labelled " +
    "series can sit in the same context as a query. TabPFN [19] is pre-trained on synthetic tabular tasks to perform " +
    "regression in context from a few labelled rows. When the inputs are fixed vectors, in-context regression can also be " +
    "learned by meta-training an embedding through a differentiable closed-form ridge solver, as in R2-D2 [17] and " +
    "MetaOptNet [18]; we use this for the MLP variant. LoRA [15] adapts a frozen network with low-rank weight updates; " +
    "rank-stabilised LoRA [16] scales them by $α/√r$ so that the learning rate need not change with the rank."),
  P("Table 2 places this work among the closest studies. The comparison with published numbers is descriptive, since " +
    "datasets and labels differ; the controlled comparison is in Section 5, where the method of Park et al. is re-run on our " +
    "data and moments."),
  ...Tab("Closest prior work. “New kind” means the test kind was not used for training.",
    ["Work", "Data used", "New kind", "Target-kind information", "Reported error"],
    [["Wang et al. [4]", "charging", "no", "training data", "0.87% (MAPE)"],
     ["Che et al. [5]", "partial charging / driving", "no", "training / fine-tuning", "1.35 / 1.15 points"],
     ["Park et al. [3]", "fleet slow charges", "—", "none (fleet P90)", "ρ > 0.80 (lab check)"],
     ["Feng et al. [6]", "charging", "yes", "unlabelled stream", "1.28–1.52; 7.95"],
     ["Chronos-2 + LoRA / ICL (ours, App. A)", "30-min clips", "yes", "$K$ labelled clips", "≥ mean-label error"],
     ["This work", "charges + rests", "yes", "$K$ labelled charges", "1.49 points ($K$ = 20)"]],
    [2150, 1600, 900, 1650, 1620]),
];

const setup = [
  H1("3  Data and evaluation"),
  P("We use six public ageing datasets, each a different cell kind (Table 3). Every recording is resampled to 10 s, " +
    "current is divided by the nominal capacity (C-rate, positive when charging) and voltage is scaled to 0–1 between the " +
    "datasheet limits, so all kinds share one scale. A check-up test gives capacity $Q_{cj}$ for cell $c$ at time $t_{cj}$; " +
    "the health label at time $t$ is"),
  Eq("y_c(t) = [(1−θ) Q_{cj} + θ Q_{c,j+1}] / max_m Q_{cm},     θ = (t − t_{cj}) / (t_{c,j+1} − t_{cj}),"),
  P("linear in time between successive check-ups and relative to the cell's highest measured capacity. The label is " +
    "retrospective: it uses later check-ups and serves for scoring only. Charges before a cell's capacity peak (break-in) " +
    "are excluded, as are clips flagged by the cleaning rules of the repository (storage, signal faults)."),
  ...Tab("Datasets. Estimation moments are the steady charges at which the method estimates health (Section 4.3).",
    ["Kind", "Cells", "Chemistry and cell", "Use between check-ups", "Health range", "Moments", "Role"],
    [["KIT [21]", "36", "NMC / C-SiO, 18650", "driving profile, 0–40 °C", "0.43–0.99", "19,800", "development"],
     ["XJTU [4]", "8", "NCM523, 18650", "random-walk discharge", "0.80–1.00", "901", "development"],
     ["CMU [22]", "22", "NCA-type, VTC6 18650", "eVTOL missions", "0.75–0.99", "2,255", "development"],
     ["Stanford LG [23]", "10", "NMC / Gr-Si, M50T 21700", "UDDS driving", "0.91–1.00", "1,218", "development"],
     ["Stanford NCA [24]", "92", "NCA / SiOx-Gr", "driving, periodic, synthetic", "0.79–1.00", "69,132", "development"],
     ["TUM [25]", "3", "NMC / Gr pouch (VW ID.3)", "highway and commuter", "0.55–0.99", "4,040", "training only"],
     ["NASA [26]", "28", "LCO, 18650", "random-walk current", "—", "—", "locked test"]],
    [1280, 560, 1640, 1640, 920, 860, 1020]),
  P("**Splits.** We evaluate leave-one-kind-out over the five development kinds: one kind is held out as the unseen kind, " +
    "and only the MLP variant is trained, on the other kinds plus TUM. Inside the held-out kind a fixed third of the cells " +
    "(seed 0) are example cells and the rest are scored cells; no cell is both. The $K$ labelled examples are charges of the " +
    "example cells, taken round-robin over them so that they are not concentrated in one cell. NASA is not touched until " +
    "the method is frozen. TUM has three cells, too few to hold out."),
  P("**Metric.** Errors are in points of health and are averaged first within each scored cell, then over cells, so that " +
    "cells with many moments do not dominate:"),
  Eq("MAE_d = (1/|C_d|) Σ_{c∈C_d} (1/N_c) Σ_{q=1}^{N_c} 100 |ŷ_{cq} − y_{cq}|,"),
  P("and the headline number is the mean of $MAE_d$ over the five kinds $d$. The reference floor is the mean of the $K$ " +
    "labels. **Error bars.** The examples are drawn 20 times per kind and $K$. For each draw we average the error over the " +
    "five kinds; figures show the median over draws and the 2.5–97.5% range. This range answers how much the result depends " +
    "on which examples happen to be labelled, the main source of variation for a user of the method. For the main " +
    "comparison we also give a paired bootstrap over scored cells (2,000 resamples of cells within each kind), which answers " +
    "whether the gain holds across cells."),
];

const methods = [
  H1("4  Method"),
  H2("4.1  Why we left the clip representation"),
  P("A 30-minute clip at 10 s is a 2 × 180 array. Chronos-2 embeddings of such clips (patch embeddings pooled per channel, " +
    "with each channel's location and scale) predict health well within a kind when trained on it (skill 0.46 on CMU, a " +
    "seen kind with a wide health range), but a ridge head trained on four kinds scores at or below the mean label on the " +
    "fifth. LoRA adaptation lowered training error while held-out-kind error rose. A transformer reader trained on " +
    "episodes whose voltage, current and labels were randomly rescaled (so that only the examples reveal the kind's " +
    "scale) still scored below the floor (skill −0.25 at $K$ = 5) and changed its output only weakly when examples were " +
    "swapped. A same-cell pilot with the Chronos-2 forecast head found that adding the clip's current and voltage to $K$ = 5 " +
    "earlier labels raised error from 5.29 to 5.68 points, and on CMU cell VAH07 the predictions stayed flat while the true " +
    "health fell. Diagnostics agreed: restricted to one health band, frozen embeddings explain almost none of the " +
    "variation within a kind (skill 0.07 on a seen kind). The bottleneck was the information in the input, not the " +
    "learner. Appendix A lists the 20 experiments behind this conclusion."),
  H2("4.2  Three label-free capacity ratios"),
  P("All three ratios compare charge now with charge for the same cell, or for a new cell of its kind, over the same " +
    "voltage interval. None uses a label of the cell."),
  P("**Window ratio.** A steady charge is a run of at least 10 min in which the current stays within 5% + 0.02C of its " +
    "median. For each cell and charge rate (within 10%), the voltage window $[v_1, v_2]$ is the widest run of 0.025-wide " +
    "voltage steps covered by at least half of the cell's charges, and must span at least 0.1 of the scale. The charge " +
    "passed while the voltage rises across the window, divided by the mean over the cell's earliest 20% of usable charges " +
    "(5 to 10 charges, taken as new), is"),
  Eq("r_c(t) = Q_{[v_1,v_2]}(t) / mean_{ref} Q_{[v_1,v_2]},     Q_{[v_1,v_2]} = ∫_{V(τ) ∈ [v_1,v_2]} I(τ) dτ."),
  P("**Fill ratio.** A full charge starts after at least 5 min below 0.02C, lasts at least 10 min and ends in the " +
    "constant-voltage taper. The charge needed to fill up from the resting voltage $V_0$ is divided by the same quantity in " +
    "the cell's earliest full charges, interpolated at $V_0$ (only within 0.05 of a reference start voltage). Because the " +
    "charge starts at rest and ends at full, growing resistance does not shift its end points."),
  P("**Rest-to-rest ratio.** A rest is at least 5 min with |I| < 0.02C; its last voltage approximates the open-circuit " +
    "voltage. Between consecutive rests with resting voltages $V_1$ and $V_2$ (|$V_2 − V_1$| ≥ 0.1) the counted charge is " +
    "$ΔQ = ∫ I dt$, whatever happened in between: driving, a discharge or a short charge. A new cell of the kind would " +
    "need $Q̄(V_1) − Q̄(V_2)$, where $Q̄(V)$ is the charge a new cell needs to fill from resting voltage $V$. We fit $Q̄$ as a " +
    "piecewise-linear function on a 0.025 voltage grid from the example cells' segments while their health is at least " +
    "0.95, separately for net-charge and net-discharge segments (the resting voltage shows hysteresis), by"),
  Eq("min_{Q̄} Σ_{segments} [Q̄(V_1) − Q̄(V_2) − ΔQ/y]^2 + μ Σ_i (Q̄_{i+1} − 2Q̄_i + Q̄_{i−1})^2,     Q̄(1) = 0,"),
  P("with μ = 1. The capacity ratio of a segment is $s = ΔQ / (Q̄(V_1) − Q̄(V_2))$. Because the curve comes from " +
    "different cells, $s$ is divided by the median of the cell's own first 30 segments before use."),
  H2("4.3  A causal feature vector at every charge"),
  P("A vehicle can estimate its health whenever a steady charge ends. At such a moment $t$, using only data recorded " +
    "before $t$, we form 15 numbers (Appendix B): the window ratio now and the median and spread of the last five; the " +
    "charge's rate, window width and temperature change relative to its reference charges; the latest fill ratio and its " +
    "age in days; the median and interquartile range of the relative rest-to-rest ratio over the last 30 net-discharge " +
    "segments, the median over the last 30 net-charge segments, and the median current variability and duration of those " +
    "drives; and two indicators for missing parts. Missing values are filled with the kind's median. Each feature is " +
    "standardised per kind with statistics of that kind's unlabelled moments, which a manufacturer would compute from its " +
    "fleet."),
  H2("4.4  In-context ridge regression"),
  P("Given the $K$ labelled examples $(x_k, y_k)$ of the new kind, we fit"),
  Eq("(a, w) = argmin Σ_{k=1}^{K} (y_k − a − w^⊤ x_k)^2 + λ ||w||^2,     ŷ_t = a + w^⊤ x_t,"),
  P("whose solution is $w = (X̃^⊤ X̃ + λI)^{−1} X̃^⊤ ỹ$ and $a = ȳ − w^⊤ x̄$, with $X̃$ and $ỹ$ centred on the examples. " +
    "The penalty λ ∈ {10^{−6}, 1, 10, 100} is chosen by leaving out one example cell at a time; only the examples are used, " +
    "so the choice adapts to each kind and $K$ without touching scored cells. Nothing is trained on other kinds. This is " +
    "in-context learning in the sense of Garg et al. [12]: the learner is fixed and the kind-specific map is inferred from " +
    "the examples at test time."),
  ...Code(["for each moment t of a scored cell:",
    "    x_t = features(record of the cell up to t)              # 15 numbers, no labels",
    "given K labelled examples (x_k, y_k) from the example cells:",
    "    for lam in {1e-6, 1, 10, 100}:",
    "        err[lam] = mean over example cells c of |ridge(examples not in c, lam)(x_c) - y_c|",
    "    a, w = ridge(all K examples, argmin err)",
    "    y_hat_t = a + w . x_t"]),
  H2("4.5  Meta-trained MLP embedding"),
  P("To test whether a learned representation helps, an MLP $φ$ (15 → 64 → 64 → 8, GELU, dropout 0.1) maps each vector to " +
    "an embedding; four of the raw features (window ratio, its five-charge median, fill ratio, drive ratio) are appended " +
    "unchanged. It is trained on the other kinds with episodes: a kind, a random third of its cells as support cells, " +
    "$K$ ~ U{5, …, 50} support moments and 128 query moments from the other cells. The prediction is the closed-form ridge " +
    "of Section 4.4 on the embeddings (λ = 1), so the loss (smooth-L1 on points of health) is differentiated through the " +
    "solver, as in R2-D2 [17]. At test time the embedding is frozen and the held-out kind is handled exactly as in Section 4.4."),
  H2("4.6  Baselines and controlled comparisons"),
  P("All methods are scored on the same moments, scored cells and example draws. (a) **Mean of the K labels**. " +
    "(b) **Single-charge ratio + line**: the window ratio alone with a least-squares line, the measurement field studies " +
    "use. (c) **Park et al. protocol + line**: our re-implementation of [3] on our data, with the latest qualifying session " +
    "before each moment (Appendix C); without a label its relative capacity is read directly as health. The comparisons " +
    "change one thing at a time: (b) versus (c) changes only the reference (the cell's own early charges versus the kind's " +
    "90th percentile); the information ladder of Section 5.2 then adds one group of features at a time with the same " +
    "learner; the MLP changes only the representation. Hyperparameters are in Table 4."),
  ...Tab("Settings. None was tuned on the scored cells; Figure 4 reports the sensitivity to the two history lengths.",
    ["Setting", "Value", "Reason"],
    [["Steady charge", "≥ 10 min, within 5% + 0.02C", "window charge is precise only over long, steady runs"],
     ["Voltage window", "widest run covered by ≥ 50% of charges, ≥ 0.1", "same window for all charges of a cell"],
     ["Reference charges", "earliest 20% of usable charges, 5–10", "enough to average; early enough to be new"],
     ["Rest", "|I| < 0.02C for ≥ 5 min", "15 min removed most segments without gain"],
     ["Rest-to-rest", "|$V_2 − V_1$| ≥ 0.1; new-cell curve from health ≥ 0.95", "avoid small denominators"],
     ["History lengths", "last 5 charges, last 30 rest-to-rest segments", "set beforehand; insensitive (Fig. 4)"],
     ["Ridge penalty", "{10^{−6}, 1, 10, 100}, leave-one-example-cell-out", "uses only the examples"],
     ["$K$, draws, example cells", "5, 10, 20, 50; 20 draws; a third of cells", "proposal design"],
     ["MLP", "64-64-8, dropout 0.1, AdamW 10^{−3}, decay 10^{−4}, 3,000 steps", "standard defaults; no validation kind to tune on"],
     ["Park et al. protocol", "CV < 25%, ≥ 5 min after 3-min trims, 10–35 °C, 0.1 V window, P90", "as published [3]"]],
    [1700, 3000, 3220], { align: AlignmentType.LEFT }),
];

const results = [
  H1("5  Results"),
  H2("5.1  Unseen kinds"),
  P("Figure 2 shows the error against the number of labelled examples. The fused vector with in-context ridge reaches " +
    "2.25 points at $K$ = 5, 1.84 at $K$ = 10 and 1.49 at $K$ = 20 (draw range 1.36–1.82). The single-charge ratio " +
    "levels off at 2.45: a line through one noisy measurement gains nothing from more examples. The protocol of Park et " +
    "al. stays near the mean label (3.80 at $K$ = 20; 6.79 points when its relative capacity is read as health without a " +
    "label). At $K$ = 50 the fused ridge is no better than at $K$ = 20 (1.55, range 1.31–2.10), probably because XJTU has " +
    "only three example cells, so the leave-one-cell-out choice of λ rests on three folds."),
  ...Fig("fig2_error_vs_k", "Error on unseen kinds against the number of labelled examples $K$. Points: median over 20 " +
    "example draws of the error averaged over the five kinds; bars: 2.5–97.5% range over draws. $K$ is on a log scale " +
    "because each step roughly doubles the labelling cost.", 5.0),
  P("Per kind (Table 5; Figure 7 in Appendix D), the gain is largest on CMU (4.05 → 1.14), where 1C charges make the " +
    "single window ratio unreliable, and on KIT (3.18 → 2.19). XJTU is the exception: its single-charge ratio is already " +
    "very precise (0.79) and the fused vector is slightly worse (0.93), because with three example cells the extra " +
    "features are fitted on very few independent cells. The paired bootstrap over scored cells gives a mean gain of 0.96 " +
    "points (95% interval 0.69–1.22): CMU +2.94, KIT +0.93, Stanford LG +0.65, Stanford NCA +0.54, XJTU −0.25."),
  ...Tab("Error per unseen kind at $K$ = 20, median over draws.",
    ["Kind", "Mean label", "Park et al. + line", "Single charge + line", "Fused, MLP", "Fused, ridge"],
    [["KIT", "5.82", "5.67", "3.18", "2.18", "**2.19**"], ["XJTU", "3.30", "3.48", "**0.79**", "0.99", "0.93"],
     ["CMU", "4.13", "5.10", "4.05", "1.27", "**1.14**"], ["Stanford LG", "2.03", "2.01", "2.19", "1.95", "**1.46**"],
     ["Stanford NCA", "3.94", "2.33", "2.07", "1.93", "**1.56**"],
     ["Mean (draw range)", "3.89", "3.80", "2.45", "1.69", "**1.49** (1.36–1.82)"]],
    [1500, 1050, 1350, 1400, 1100, 1520]),
  H2("5.2  Which information helps"),
  P("Figure 3 adds one group of features at a time to the single-charge ratio, with the same learner and moments. The " +
    "history of the last five ratios alone does not help (2.48); the charge conditions lower the error to 2.13 and the " +
    "latest full charge to 1.97, both with wide draw ranges, because more features fitted on 20 examples are sometimes " +
    "fitted badly. Adding the rest-to-rest ratios of the drives between charges lowers the error to 1.60 and also narrows " +
    "the range; the rest-to-rest charges bring it to 1.49. Adding the reading of Park et al. as a 16th feature changes " +
    "little (1.46). The between-charge measurements are thus the largest single contribution."),
  ...Fig("fig3_information_ladder", "Information ladder at $K$ = 20. Each bar adds one group of features to the one " +
    "above, with the same in-context ridge and moments. Bars: mean over the five kinds (median over draws), with the " +
    "2.5–97.5% draw range; open circles: each kind's median."),
  H2("5.3  How many measurements are needed"),
  P("The two history lengths were fixed in advance (five charges, 30 segments). Figure 4 varies them. A single " +
    "rest-to-rest segment already gives 1.55 points and 10–30 give 1.49–1.51; using 100 raises the error to 1.59, because " +
    "old segments describe an older cell. One past charge gives 1.55 and three to ten give 1.49–1.51. A handful of recent " +
    "measurements is enough."),
  ...Fig("fig5_measurement_counts", "Sensitivity to the history lengths at $K$ = 20 (fused ridge). Left: number of recent " +
    "rest-to-rest segments summarised, on a log scale since the values span two decades. Right: number of past charge " +
    "ratios. Points and bars as in Figure 2."),
  H2("5.4  The learned embedding"),
  P("Figure 5 shows the meta-trained MLP. Its error on episodes of the training kinds falls from 1.3 to 0.8 points over " +
    "3,000 steps, while its error on the held-out kind stays near 1.35 throughout. With four training kinds per fold, the " +
    "embedding learns what separates those kinds rather than a map that transfers. Evaluated like the ridge, it reaches " +
    "1.69 at $K$ = 20, between the single-charge ratio and the fused ridge."),
  ...Fig("fig6_mlp_training", "Meta-MLP training curves, one thin line per held-out kind and the mean in bold. Training " +
    "MAE: queries of the training episodes, averaged over 100 steps. Held-out MAE: ridge (λ = 1) on 20 examples of the " +
    "held-out kind, evaluated on 2,000 of its moments; it is monitored only and selects nothing.", 5.0),
  H2("5.5  One cell over time, and a control"),
  P("Figure 6 follows one CMU cell. The single-charge ratio falls into two bands, one per charge rate, and follows the " +
    "decline only loosely. The fused estimate follows the decline but shows a sawtooth that repeats between check-ups; " +
    "we have not yet identified its cause, and removing it is a direction for the final version."),
  ...Fig("fig7_trajectory", "Health estimates for the CMU scored cell with the most moments, $K$ = 20, one example draw.", 5.0),
  P("**Control for a laboratory artefact.** In a laboratory test a fixed drive profile ends sooner as the cell ages, so " +
    "drive duration could reveal health in a way a real vehicle would not. Drive duration alone scores at the floor " +
    "(KIT 5.52 against 5.82), and removing it from the vector leaves the mean error unchanged (1.44 against 1.45 in the " +
    "same run), so the gain does not come from it."),
];

const discussion = [
  H1("6  Discussion"),
  P("**What we learned.** The transfer problem in our proposal was posed at the wrong level. A 30-minute clip mixes the " +
    "effects of load, state of charge, temperature and age, and with five training kinds no model we tried separated age " +
    "from the rest. Measurements that are ratios to the same cell's early life, or to a new cell of its kind, remove most " +
    "of the kind-specific scale before any learning, and what remains is close to linear, so 20 labelled examples suffice " +
    "to fit it. The gain over a single charge comes mainly from measurements taken between charges, which are free in any " +
    "vehicle that rests, and from the conditions under which each charge was taken. Comparing each cell with its own early " +
    "life rather than with the fleet is the largest single step (3.80 → 2.45)."),
  P("**Why ridge and not a network.** The learned embedding fitted the training kinds and did not transfer, and ridge " +
    "on the raw vector was better on four of five kinds. With five kinds, the space of kinds is too thin to learn a " +
    "representation from. A model pre-trained on many synthetic regression tasks, such as TabPFN [19], uses the same " +
    "examples without learning from our kinds and is the next learner we will test."),
  P("**Limitations.** All data come from laboratory tests. Chambers hold the temperature constant, so the seasonal " +
    "effect reported by Park et al. [3] is not tested. Laboratory charges are cleaner and more frequent than a driver's, " +
    "and the reference charges of each cell are its first ones, which in a vehicle are the first weeks of use. The " +
    "comparison with Park et al. uses their protocol on cells tested under several conditions per kind, which is less " +
    "favourable to a fleet-percentile reference than a fleet of identical vehicles. The method of Feng et al. [6] is " +
    "cited, not re-run. Labels are interpolated between check-ups. XJTU has only three example cells."),
  P("**In progress for the final report.** (1) The locked NASA test. (2) Label-shuffle and swapped-kind controls for the " +
    "in-context ridge (hypothesis H4). (3) Realistic charge schedules: thinning the charges to a weekly partial charge to " +
    "measure how much the between-charge measurements matter when charges are rare. (4) TabPFN as the in-context learner."),
];

const refs = [
  "California Air Resources Board. Durability requirements in the Advanced Clean Cars II program. Presentation EVE-57-13e to the UNECE EVE working group, 2022.",
  "Regulation (EU) 2023/1542 of the European Parliament and of the Council concerning batteries and waste batteries, 2023.",
  "J. Park, K. Kim, S. Geum, J. Lee, H. Son, S. Han. Battery health reporting fails independent validation across manufacturers. arXiv:2603.21592, 2026.",
  "F. Wang, Z. Zhai, Z. Zhao, Y. Di, X. Chen. Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis. Nature Communications 15:4332, 2024.",
  "Y. Che, V. N. Lam, J. Rhyu, J. Schaeffer, M. Kim, M. Z. Bazant, W. C. Chueh, R. D. Braatz. Diagnostic-free onboard battery health assessment. arXiv:2503.07383, 2025.",
  "Y. Feng, G. Hu, X. Li, Z. Zhang. Adapting amidst degradation: cross domain Li-ion battery health estimation via physics-guided test-time training. arXiv:2402.00068, 2024.",
  "A. Lanubile, P. Bosoni, G. Pozzato, A. Allam, M. Acquarone, S. Onori. Domain knowledge-guided machine learning framework for state of health estimation in lithium-ion batteries. Communications Engineering 3:168, 2024.",
  "Y. Chen, Q. Hou, J. Liu, Y. Wang. Reference selection for cross-condition lithium-ion battery state-of-health estimation. World Electric Vehicle Journal 17(11):582, 2026.",
  "R. R. Richardson, M. A. Osborne, D. A. Howey. Battery health prediction under generalized conditions using a Gaussian process transition model. Journal of Energy Storage 23:320–328, 2019.",
  "R. Tan et al. BatteryLife: a comprehensive dataset and benchmark for battery life prediction. KDD, 2025.",
  "W. Sun et al. Fine-tuning enables state of health estimation for lithium-ion batteries via a time series foundation model. Energy 318:134177, 2025.",
  "S. Garg, D. Tsipras, P. Liang, G. Valiant. What can transformers learn in-context? A case study of simple function classes. NeurIPS, 2022.",
  "A. F. Ansari et al. Chronos: learning the language of time series. Transactions on Machine Learning Research, 2024.",
  "A. F. Ansari et al. Chronos-2: from univariate to universal forecasting. arXiv:2510.15821, 2025.",
  "E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. LoRA: low-rank adaptation of large language models. ICLR, 2022.",
  "D. Kalajdzievski. A rank stabilization scaling factor for fine-tuning with LoRA. arXiv:2312.03732, 2023.",
  "L. Bertinetto, J. F. Henriques, P. H. S. Torr, A. Vedaldi. Meta-learning with differentiable closed-form solvers. ICLR, 2019.",
  "K. Lee, S. Maji, A. Ravichandran, S. Soatto. Meta-learning with differentiable convex optimization. CVPR, 2019.",
  "N. Hollmann et al. Accurate predictions on small data with a tabular foundation model. Nature 637:319–326, 2025.",
  "G. L. Plett. Recursive approximate weighted total least squares estimation of battery cell total capacity. Journal of Power Sources 196(4):2319–2331, 2011.",
  "M. Luh, T. Blank. Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell. Scientific Data 11:1004, 2024.",
  "A. Bills, S. Sripad, L. Fredericks, M. Guttenberg, D. Charles, E. Frank, V. Viswanathan. A battery dataset for electric vertical takeoff and landing aircraft. Scientific Data 10:344, 2023.",
  "G. Pozzato, A. Allam, S. Onori. Lithium-ion battery aging dataset based on electric vehicle real-driving profiles. Data in Brief 41:107995, 2022.",
  "A. Geslin et al. Dynamic cycling enhances battery lifetime. Nature Energy 10:172–180, 2025.",
  "M. Schreiber et al. Understanding lithium-ion battery degradation in vehicle applications: insights from realistic and accelerated aging tests using Volkswagen ID.3 pouch cells. Journal of Energy Storage 112:115357, 2025.",
  "C. Bole, C. Kulkarni, M. Daigle. Adaptation of an electrochemistry-based Li-ion battery model to account for deterioration observed under randomized use. PHM Conference, 2014.",
  "M. Dubarry, V. Svoboda, R. Hwu, B. Y. Liaw. Incremental capacity analysis and close-to-equilibrium OCV measurements to quantify capacity fade in commercial rechargeable lithium batteries. Electrochemical and Solid-State Letters 9(10):A454, 2006.",
  "I. Bloom et al. Differential voltage analyses of high-power, lithium-ion cells: 1. Technique and application. Journal of Power Sources 139:295–303, 2005.",
  "K. A. Severson et al. Data-driven prediction of battery cycle life before capacity degradation. Nature Energy 4:383–391, 2019.",
  "D. Roman, S. Saxena, V. Robu, M. Pecht, D. Flynn. Machine learning pipeline for battery state-of-health estimation. Nature Machine Intelligence 3:447–456, 2021.",
  "US Patent 10,371,753. Methods for online estimation of battery capacity and state of health.",
];
const references = [H1("References"),
  ...refs.map((r, k) => new Paragraph({ children: runs(`[${k + 1}]  ${r}`, 17), spacing: { after: 50 },
    indent: { left: 360, hanging: 360 }, alignment: AlignmentType.LEFT }))];

const appendix = [
  new Paragraph({ pageBreakBefore: true, children: [new TextRun({ text: "Appendix", font: FONT, size: 28, bold: true })], spacing: { after: 120 } }),
  H1("A  Everything else we tried"),
  P("Experiments in order. Skill = 1 − MAE / MAE of the matching mean-label guess (0: no better than the guess). Numbers " +
    "come from different scoring sets as the work evolved and are comparable within a row, not across rows."),
  ...Tab("Exploratory experiments. Scripts and results for all of them are kept in the project record.",
    ["Experiment", "Question", "Outcome"],
    [["Frozen Chronos-2 + ridge, CMU (seen kind)", "Do embeddings carry health?", "Yes on a wide health range: 2.66 vs 4.94 (skill 0.46)"],
     ["Same-cell pilot, Chronos-2 forecast head", "Does the clip add to $K$ earlier labels of the same cell?", "No: 5.68 with current and voltage vs 5.29 without ($K$ = 5)"],
     ["Frozen Chronos-2 + ridge, unseen kind", "Transfer with no examples?", "Skill ≤ 0 on every kind"],
     ["Chronos-2 forecast head reading $K$ examples", "Native in-context use of other cells?", "No setting beat the mean label"],
     ["Data scaling, 25/100/400 clips per kind", "Does more training data help transfer?", "No mode beat the mean label at any size (stopped early)"],
     ["LoRA rank 8 / 32 / 128 (rank-stabilised)", "Does adapter size matter?", "Held-out error rose while training error fell"],
     ["Learned in-context reader, disguised episodes", "Can a reader be forced to use examples?", "Skill −0.25 at $K$ = 5; weak response to swapped examples"],
     ["Physics features vs Chronos-2 embeddings", "Which clip representation carries health?", "Seen kind: 0.25 vs 0.07"],
     ["In-context reader on physics features", "Better inputs for the reader?", "−0.12 (disguised) vs −0.23; still below the floor"],
     ["Model fitted on $K$ examples only", "Do other cells of a kind inform a new cell?", "−0.02 even at $K$ = 50"],
     ["Relative and dQ/dV features", "Features relative to the cell's early life?", "Seen kind 0.34; unseen positive on 3 of 5 kinds"],
     ["Examples set the level, model the slope", "Anchoring on the $K$ labels", "Skill +0.08–0.11, 95% interval above 0, on 16-clip windows"],
     ["Window ratio from clips and full records", "Health from steady charging, no labels", "Tracks ageing (XJTU r = 0.97); CMU fails at 1C"],
     ["Fill ratio from full charges", "Resistance-free charge measurement", "XJTU 0.99 with no labels"],
     ["Examples choose measurement and smoothing", "In-context selection among ratios", "1.97 mean with every cell scored (1.61 on covered cells only)"],
     ["Rest-to-rest ratio, driving only", "Health with no charging at all", "CMU 2.64, KIT 3.97, LG 1.50; NCA fails (4.94)"],
     ["Availability of usable charges", "Are clean charges rare in our data?", "No: nearly every cell has them"],
     ["Fused vector (this paper)", "Charge plus between-charge measurements", "1.49 at $K$ = 20 (Section 5)"]],
    [2500, 2600, 2820], { align: AlignmentType.LEFT }),
  H1("B  The 15 features"),
  ...Tab("Features at moment $t$; all use data before $t$ only and no labels.",
    ["Group", "Feature", "Definition"],
    [["Charge", "est", "window ratio $r_c$ of the current charge"],
     ["", "est_med5, est_std5", "median and standard deviation of the last 5 window ratios"],
     ["Conditions", "rate, span, delta_temp", "charge C-rate; window width; mean temperature minus that of the reference charges"],
     ["Full charge", "full_est, full_age_days, has_full", "median of the last 3 fill ratios; days since; indicator"],
     ["Between charges", "drive_med, drive_iqr", "median and IQR of relative $s$ over the last 30 net-discharge segments"],
     ["", "charge_seg_med", "median relative $s$ over the last 30 net-charge segments"],
     ["", "drive_current_std, drive_hours, has_drive", "median current variability and duration of those drives; indicator"]],
    [1500, 2700, 3720], { align: AlignmentType.LEFT }),
  H1("C  Re-implementation of Park et al."),
  P("Following the methods of [3]: in each charge, the constant-current segment is the longest run within 25% of the " +
    "charge's median current; it must have a current coefficient of variation below 25%, last at least 5 min after " +
    "trimming 3 min at each end, rise in voltage, and have a mean temperature between 10 and 35 °C where temperature is " +
    "recorded. The charge passed in a fixed 0.1 V window is measured; the window is chosen per kind to cover the most " +
    "sessions (KIT 3.68–3.78 V, XJTU 3.73–3.83 V, CMU 3.88–3.98 V, Stanford LG 4.09–4.19 V, Stanford NCA 3.80–3.90 V), " +
    "and normalised by the kind's 90th percentile. Qualifying sessions: KIT 16,835, XJTU 152, CMU 2,664, Stanford LG " +
    "1,540, Stanford NCA 139,079. Applying the 25% rule to the whole charge, including the constant-voltage taper, " +
    "rejected almost all sessions on CMU and XJTU in a first run; the segment rule above follows the paper."),
  H1("D  Per-kind results and reproduction"),
  ...Fig("fig4_per_kind", "Error per unseen kind at $K$ = 20: median over draws with the 2.5–97.5% range.", 5.0),
  P("The repository (github.com/zivbeh/In-Context-Battery-Health-Estimation-for-Unseen-Battery-Kinds) fetches every " +
    "dataset with scripts/download_data.sh and standardises it with scripts/preprocess.py. The measurements are produced " +
    "by scripts/run_x15_charge_events.py (window ratios), run_x16_full_charges.py (fill ratios) and run_x24_rest_to_rest.py " +
    "(rest-to-rest segments); scripts/run_x29_paper_results.py runs every method of Sections 4–5 and writes the per-draw " +
    "results; figures/make_figures.py draws all figures and writes every table value to figures/tables.md without a GPU."),
  H1("E  Use of AI tools"),
  P("Claude (Anthropic) was used as a coding and writing assistant: to write and debug analysis scripts, run experiments " +
    "on our GPU machine, search the literature and draft text. All experimental choices, results and claims were checked " +
    "by the authors against the code and the result files; one bug the assistant introduced (a feature group leaking " +
    "into every rung of the information ladder) was found during this check and fixed before the numbers above were " +
    "produced. Citations were checked against the papers in the project folder."),
];

// ------------------------------------------------------------------ assemble
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, bottom: 1440, left: 2160, right: 2160 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: [...title, abstract, ...intro, ...methodFig,
      ...related, ...setup, ...methods, ...results, ...discussion, ...references, ...appendix],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "report_draft.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
