"""
Generate SUBMISSION.pdf for the Deloitte Quantum Sustainability Challenge.
Run: python3 generate_submission_pdf.py
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ─── Colours ────────────────────────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#1A2F5E")
MID_BLUE    = colors.HexColor("#2E5B9A")
LIGHT_BLUE  = colors.HexColor("#EAF0FB")
ACCENT_GOLD = colors.HexColor("#C8A535")
LIGHT_GRAY  = colors.HexColor("#F5F5F5")
MID_GRAY    = colors.HexColor("#CCCCCC")
TEXT_BLACK  = colors.HexColor("#1A1A1A")

# ─── Styles ─────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def make_styles():
    s = {}
    s["title"] = ParagraphStyle("title",
        fontName="Helvetica-Bold", fontSize=20,
        textColor=DARK_BLUE, alignment=TA_CENTER,
        spaceAfter=6, leading=24)
    s["subtitle"] = ParagraphStyle("subtitle",
        fontName="Helvetica", fontSize=13,
        textColor=MID_BLUE, alignment=TA_CENTER,
        spaceAfter=4, leading=16)
    s["tagline"] = ParagraphStyle("tagline",
        fontName="Helvetica-Oblique", fontSize=10,
        textColor=colors.HexColor("#555555"), alignment=TA_CENTER,
        spaceAfter=2)
    s["section"] = ParagraphStyle("section",
        fontName="Helvetica-Bold", fontSize=13,
        textColor=DARK_BLUE, spaceBefore=14, spaceAfter=6,
        leading=16)
    s["subsection"] = ParagraphStyle("subsection",
        fontName="Helvetica-Bold", fontSize=11,
        textColor=MID_BLUE, spaceBefore=10, spaceAfter=4,
        leading=13)
    s["subsubsection"] = ParagraphStyle("subsubsection",
        fontName="Helvetica-Bold", fontSize=10,
        textColor=TEXT_BLACK, spaceBefore=6, spaceAfter=3,
        leading=12)
    s["body"] = ParagraphStyle("body",
        fontName="Helvetica", fontSize=9.5,
        textColor=TEXT_BLACK, alignment=TA_JUSTIFY,
        spaceAfter=5, leading=14)
    s["body_center"] = ParagraphStyle("body_center",
        fontName="Helvetica", fontSize=9.5,
        textColor=TEXT_BLACK, alignment=TA_CENTER,
        spaceAfter=5, leading=14)
    s["bullet"] = ParagraphStyle("bullet",
        fontName="Helvetica", fontSize=9.5,
        textColor=TEXT_BLACK, leftIndent=14,
        spaceAfter=3, leading=13)
    s["cell"] = ParagraphStyle("cell",
        fontName="Helvetica", fontSize=8.5,
        textColor=TEXT_BLACK, leading=11)
    s["cell_bold"] = ParagraphStyle("cell_bold",
        fontName="Helvetica-Bold", fontSize=8.5,
        textColor=TEXT_BLACK, leading=11)
    s["cell_center"] = ParagraphStyle("cell_center",
        fontName="Helvetica", fontSize=8.5,
        textColor=TEXT_BLACK, alignment=TA_CENTER, leading=11)
    s["cell_bold_center"] = ParagraphStyle("cell_bold_center",
        fontName="Helvetica-Bold", fontSize=8.5,
        textColor=TEXT_BLACK, alignment=TA_CENTER, leading=11)
    s["note"] = ParagraphStyle("note",
        fontName="Helvetica-Oblique", fontSize=8.5,
        textColor=colors.HexColor("#555555"),
        spaceAfter=4, leading=11)
    s["repo"] = ParagraphStyle("repo",
        fontName="Courier-Bold", fontSize=9,
        textColor=MID_BLUE, alignment=TA_CENTER,
        spaceAfter=4, leading=12)
    return s

S = make_styles()

def sp(n=6):
    return Spacer(1, n)

def hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=MID_GRAY, spaceAfter=4, spaceBefore=4)

def section(text):
    return Paragraph(text, S["section"])

def subsection(text):
    return Paragraph(text, S["subsection"])

def subsubsection(text):
    return Paragraph(text, S["subsubsection"])

def body(text):
    return Paragraph(text, S["body"])

def bullet(text):
    return Paragraph(f"&bull;&nbsp;&nbsp;{text}", S["bullet"])

def note(text):
    return Paragraph(text, S["note"])

# ─── Table builders ──────────────────────────────────────────────────────────
HDR_STYLE = [
    ("BACKGROUND",  (0,0), (-1,0), DARK_BLUE),
    ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
    ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",    (0,0), (-1,0), 8.5),
    ("BOTTOMPADDING",(0,0),(-1,0), 5),
    ("TOPPADDING",  (0,0), (-1,0), 5),
    ("GRID",        (0,0), (-1,-1), 0.4, MID_GRAY),
    ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
    ("FONTSIZE",    (0,1), (-1,-1), 8.5),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT_GRAY]),
    ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",  (0,1), (-1,-1), 4),
    ("BOTTOMPADDING",(0,1),(-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING",(0,0), (-1,-1), 5),
]

def make_table(data, col_widths, extra_style=None):
    style = list(HDR_STYLE)
    if extra_style:
        style.extend(extra_style)
    return Table(data, colWidths=col_widths,
                 style=TableStyle(style), repeatRows=1)

def p(text, bold=False, center=False):
    if bold and center:
        return Paragraph(text, S["cell_bold_center"])
    if bold:
        return Paragraph(text, S["cell_bold"])
    if center:
        return Paragraph(text, S["cell_center"])
    return Paragraph(text, S["cell"])

# ─── Build document ──────────────────────────────────────────────────────────
W = 6.5 * inch   # usable width (letter minus 1" margins each side)

def build_pdf(outfile="SUBMISSION.pdf"):
    doc = SimpleDocTemplate(
        outfile,
        pagesize=letter,
        leftMargin=inch, rightMargin=inch,
        topMargin=0.9*inch, bottomMargin=0.9*inch,
        title="Deloitte Quantum Sustainability Challenge — GenQ Submission",
        author="Priti Sagar, Gautam Mahajan",
    )
    story = []

    # ── Cover / header block ────────────────────────────────────────────────
    story += [
        sp(8),
        Paragraph("Deloitte Quantum Sustainability Challenge 2026", S["title"]),
        Paragraph("Team GenQ — Phase 1 Submission", S["subtitle"]),
        Paragraph(
            "Hybrid Quantum-Classical Wildfire Risk Prediction &amp; Insurance Premium Estimation",
            S["tagline"]),
        sp(4),
        HRFlowable(width="100%", thickness=2, color=ACCENT_GOLD,
                   spaceAfter=6, spaceBefore=0),
        sp(4),
    ]

    # ── Section 1: Team Overview ─────────────────────────────────────────────
    story += [section("Section 1: Team Overview"), hr()]

    team_data = [
        [p("Team Name", bold=True), p("GenQ", bold=True)],
        [p("Member 1"),  p("Priti Sagar")],
        [p("University"), p("Drexel University")],
        [p("Email"),     p("pp693@drexel.edu")],
        [p("Member 2"),  p("Gautam Mahajan")],
        [p("University"), p("University of Connecticut – Storrs")],
        [p("Email"),     p("gautam.mahajan@uconn.edu")],
    ]
    t = Table(team_data, colWidths=[2.2*inch, 4.3*inch],
              style=TableStyle([
                  ("GRID",        (0,0), (-1,-1), 0.4, MID_GRAY),
                  ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
                  ("FONTSIZE",    (0,0), (-1,-1), 9.5),
                  ("ROWBACKGROUNDS",(0,0),(-1,-1), [LIGHT_BLUE, colors.white]),
                  ("TOPPADDING",  (0,0), (-1,-1), 4),
                  ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                  ("LEFTPADDING", (0,0), (-1,-1), 6),
                  ("RIGHTPADDING",(0,0), (-1,-1), 6),
                  ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
              ]))
    story += [t, sp(6)]

    # ── Section 2: Abstract ──────────────────────────────────────────────────
    story += [section("Section 2: Abstract"), hr()]

    abstract_paras = [
        ("We present a hybrid quantum-classical system for California wildfire risk prediction "
         "and insurance premium estimation, built in three validated layers."),

        ("<b>Layer 1 — Q-MoE Fire (regime-diverse quantum experts):</b> California's 58 counties "
         "cluster into 4 climate regimes with fundamentally different fire dynamics. Q-MoE routes "
         "each county-month through 4 quantum reservoir experts, each using a different entanglement "
         "topology (ring, ladder, star, full). The concatenated 44-dimensional feature vector feeds "
         "a GradientBoosting head. Q-MoE achieves F1=0.782 and AUC=0.934, outperforming SVM "
         "(F1=0.709) by 10.3%. Per-regime ablation confirms improvements in all 4 climate zones."),

        ("<b>Layer 2 — RA-MQTF (multi-task fusion):</b> RA-MQTF adds a shared multi-task neural "
         "backbone on top of Q-MoE's expert features, jointly predicting fire occurrence, severity, "
         "and insurance premiums. Multi-task training improves fire classification to F1=0.834 "
         "(+6.6% over Q-MoE) and severity to R2=0.639 (+19.4%). The premium task via county-to-ZIP "
         "mapping confirms that the county granularity is the binding constraint — exactly the gap "
         "ZIP-STR-QMoE (Section 5) addresses directly."),

        ("<b>Layer 3 — QLSTM (temporal premium model):</b> 4-qubit variational LSTM gates achieve "
         "R2=0.349 on temporal log-scale splits, substantially outperforming Classical LSTM "
         "(R2=−1.033). With only 4 years of annual ZIP data available, QLSTM's advantage "
         "demonstrates that quantum gates capture temporal premium dynamics that standard "
         "recurrent cells miss."),

        ("<b>Main limitation:</b> Q-MoE and RA-MQTF operate at county granularity; ZIP-level "
         "predictions use a county-to-ZIP geographic mapping. The proposed ZIP-STR-QMoE algorithm "
         "(Section 5) addresses this directly."),

        ("<b>Forward-looking predictions:</b> Per-ZIP wildfire risk scores for 2,174 California "
         "locations, combining Q-MoE county fire probabilities with domain-driven indicators."),

        ("<b>Proposed algorithm:</b> ZIP-STR-QMoE (Section 5) extends these validated ingredients "
         "to direct ZIP-level prediction, addressing the challenge's core granularity requirement."),
    ]
    for txt in abstract_paras:
        story.append(body(txt))
    story.append(sp(4))

    # ── Section 3: Detailed Algorithm Description ────────────────────────────
    story += [section("Section 3: Detailed Algorithm Description"), hr()]

    # 3.1 Data Pipeline
    story += [subsection("3.1  Data Pipeline"), sp(2)]

    story.append(body("<b>Challenge-provided datasets:</b>"))
    story += [
        bullet("<b>Wildfire data:</b> California wildfire records with county, burned acreage "
               "(GIS_ACRES), and weather. Monthly county-level granularity — 10,989 rows, "
               "58 counties, 2008–2020."),
        bullet("<b>Insurance data:</b> California Department of Insurance homeowners filings — "
               "<i>insurance_2018_2019.XLS</i> and <i>insurance_2020_2021.XLS</i> — containing "
               "ZIP-level premiums, exposures, losses, and risk scores (~2,000 ZIPs/year)."),
        sp(4),
    ]

    story.append(body(
        "<b>Additional data used: None.</b> All analysis uses only the five "
        "challenge-provided files:"))
    story += [
        bullet("<i>wildfire_county_monthly.csv</i> — monthly county-level wildfire and weather "
               "records (2008–2020)"),
        bullet("<i>wildfire_weather_daily.csv</i> — daily weather and fire occurrence records, "
               "used for temporal trend analysis and 2026 forward projections"),
        bullet("<i>insurance_2018_2019.XLS</i> and <i>insurance_2020_2021.XLS</i> — ZIP-level "
               "homeowners insurance premiums and risk scores"),
        bullet("<i>geojson-counties-fips.json</i> — California county boundaries for "
               "geospatial visualisation"),
        sp(2),
    ]

    story.append(note("No external APIs, third-party datasets, or real-time data feeds were used."))

    story.append(body(
        "<b>Feature engineering:</b> Temperature range, wind–temperature ratio, lagged "
        "precipitation and wind speed, PCA-reduced quantum encodings (4 components), county "
        "adjacency graph features, temporal lag features (lag_acres, lag_temp), and "
        "domain-driven indicators (Fire Weather Index proxy, drought severity, hydroclimate "
        "whiplash signals)."))

    story.append(body(
        "<b>Temporal split:</b> Train 2008–2018 / test 2019–2020 (county wildfire); "
        "train 2018–2020 / test 2021 (ZIP insurance). "
        "<i>Note: the provided wildfire_county_monthly.csv ends at 2020-12. "
        "Q-MoE and RA-MQTF use 2008–2018 training to preserve a held-out 2019–2020 test set. "
        "For the challenge's 2018–2021 window, the insurance dataset is used in full for Task 2, "
        "and the daily weather data (extending to 2025) supplements county wildfire data for "
        "trend analysis and 2026 forward projections.</i>"))
    story.append(sp(4))

    # 3.2 Task 1A
    story += [subsection("3.2  Task 1A: Wildfire Classification"), sp(2)]

    story.append(body(
        "<b>Primary model: Q-MoE Fire — Quantum Mixture-of-Experts</b>  "
        "(<i>04_proposed_algorithm/01_QMoE_Fire</i>)"))

    story.append(body(
        "California spans 16 climate zones with fundamentally different fire dynamics. "
        "A single quantum model forced to learn all regimes simultaneously must average "
        "across them. Q-MoE instead routes each county-month to specialised quantum "
        "experts matched to its climate regime."))

    story += [subsubsection("Architecture"), sp(2)]

    arch_steps = [
        ("<b>1. Regime discovery:</b> KMeans clustering on county-level geography and weather "
         "produces 4 natural climate regimes:<br/>"
         "&nbsp;&nbsp;&bull; Regime A (24 counties): Moderate coastal/central California — 4.9% fire rate<br/>"
         "&nbsp;&nbsp;&bull; Regime B (13 counties): Hot, fire-prone inland (Fresno, Kern, L.A.) — 39% fire rate<br/>"
         "&nbsp;&nbsp;&bull; Regime C (17 counties): Cool, wet northern (Humboldt, Shasta, Butte) — 20% fire rate<br/>"
         "&nbsp;&nbsp;&bull; Regime D (4 counties): Extreme desert (Imperial, Inyo, Riverside, San Bernardino) — 24% fire rate"),
        ("<b>2. Four quantum reservoir experts:</b> Each expert uses 4 qubits and 10 reservoir "
         "layers with fixed (non-trainable) random weights and genuinely different entanglement "
         "topologies — ring (circular CNOT chain), ladder (alternating even/odd pairs), star "
         "(hub-and-spoke), and full (all-to-all). Each expert produces 11 observables "
         "(4 PauliZ + 4 PauliX + 3 nearest-neighbour ZZ correlations)."),
        ("<b>3. Concat gating:</b> All 44 expert features (4×11) are concatenated and fed to a "
         "GradientBoosting head that learns which expert features matter for each sample. "
         "This outperformed explicit hard gating and soft gating."),
        ("<b>Result: F1 = 0.782, AUC = 0.934</b>"),
    ]
    for txt in arch_steps:
        story.append(bullet(txt))
    story.append(sp(4))

    story.append(body(
        "<b>Multi-task improvement — RA-MQTF</b>  (<i>04_proposed_algorithm/04_RegimeAware_MQTF</i>):  "
        "Adding a shared multi-task backbone on top of Q-MoE expert features further improves "
        "fire classification to <b>F1 = 0.834</b> (+6.6%) and fire severity to "
        "<b>R2 = 0.639</b> (+19.4%). Joint training acts as regularisation."))
    story.append(sp(4))

    story += [subsubsection("Per-Regime Validation — Q-MoE improves all 4 climate zones"), sp(2)]
    regime_data = [
        [p("Regime", bold=True,center=True), p("N test",bold=True,center=True),
         p("Fire Rate",bold=True,center=True), p("MoE F1",bold=True,center=True),
         p("Baseline F1",bold=True,center=True), p("F1 Lift",bold=True,center=True)],
        [p("A — coastal",center=True),  p("570",center=True), p("7.4%",center=True),
         p("0.693",center=True), p("0.582",center=True), p("+0.111",center=True)],
        [p("B — inland hot",center=True), p("308",center=True), p("41.9%",center=True),
         p("0.833",center=True), p("0.736",center=True), p("+0.098",center=True)],
        [p("C — northern wet",center=True), p("402",center=True), p("22.9%",center=True),
         p("0.739",center=True), p("0.680",center=True), p("+0.059",center=True)],
        [p("D — desert",center=True), p("94",center=True), p("24.5%",center=True),
         p("0.818",center=True), p("0.708",center=True), p("+0.110",center=True)],
    ]
    story.append(make_table(regime_data,
        [1.3*inch, 0.7*inch, 0.9*inch, 0.9*inch, 1.0*inch, 0.85*inch]))
    story.append(sp(6))

    story += [subsubsection("All Quantum Classifiers Explored (identical test splits)"), sp(2)]
    qcls_data = [
        [p("Model",bold=True), p("F1",bold=True,center=True),
         p("Qubits",bold=True,center=True), p("Notebook",bold=True)],
        [p("RA-MQTF (multi-task)",bold=True), p("0.834",bold=True,center=True),
         p("4×4",center=True), p("04_proposed_algorithm/04_RegimeAware_MQTF")],
        [p("Q-MoE Concat + GBM"), p("0.782",center=True), p("4×4",center=True),
         p("04_proposed_algorithm/01_QMoE_Fire")],
        [p("QART (cross-attention)"), p("0.755",center=True), p("4",center=True),
         p("04_proposed_algorithm/02_QART")],
        [p("QRC + LogReg"), p("0.724",center=True), p("8",center=True),
         p("02_quantum_baselines/03_Cloud_Quantum_Models")],
        [p("MQTF (multi-task)"), p("0.693",center=True), p("4",center=True),
         p("03_novel_explorations/04_MultiTask_QTF")],
        [p("QGR (graph)"), p("0.650",center=True), p("4",center=True),
         p("03_novel_explorations/02_Quantum_Graph_Reservoir")],
        [p("Dressed Quantum Circuit"), p("0.637",center=True), p("4",center=True),
         p("02_quantum_baselines/04_Advanced_Architectures")],
        [p("VQC (variational)"), p("0.542",center=True), p("6",center=True),
         p("01_eda_and_visualization/01_EDA_and_Baselines")],
    ]
    story.append(make_table(qcls_data,
        [2.3*inch, 0.7*inch, 0.7*inch, 2.8*inch]))
    story.append(sp(8))

    # 3.3 Task 1B
    story += [subsection("3.3  Task 1B: Evaluation"), sp(2)]

    story.append(body("<b>Advantages of our quantum approach:</b>"))
    story += [
        bullet("<b>No barren plateaus:</b> Fixed quantum reservoirs (QRC, Q-MoE) have zero "
               "trainable quantum parameters, completely avoiding the barren plateau problem "
               "that undermines variational methods (VQC achieved only F1=0.542)."),
        bullet("<b>Regime specialisation:</b> Different entanglement topologies create diverse "
               "feature maps capturing different fire dynamics — validated by per-regime "
               "improvements in all 4 climate zones."),
        bullet("<b>NISQ-ready:</b> 4-qubit reservoirs at depth 40 are deployable on current "
               "hardware (IonQ Aria 25-qubit, Rigetti Ankaa-3). AWS Braket integration is "
               "implemented in the Cloud Quantum Models notebook."),
        bullet("<b>Interpretable:</b> Classical GBM and LogReg heads provide feature importance; "
               "SHAP analysis identifies key wildfire drivers."),
        sp(4),
    ]

    story.append(body("<b>Disadvantages and limitations:</b>"))
    story += [
        bullet("<b>Quantum encoding overhead:</b> Encoding 8,584 samples through 4 experts "
               "takes ~7 minutes on a simulator with no parallelisation advantage over "
               "classical GBM."),
        bullet("<b>Indirect quantum benefit:</b> The quantum reservoir is a non-linear feature "
               "extractor; classification is performed by a classical head. Benefit comes from "
               "the feature space, not quantum speedup."),
        bullet("<b>County granularity:</b> Q-MoE and RA-MQTF operate at county level; "
               "ZIP predictions require a mapping step that reduces spatial precision."),
        sp(4),
    ]

    story.append(body("<b>Classical comparison (identical data, identical splits):</b>"))
    cls_cmp_data = [
        [p("Model",bold=True), p("F1",bold=True,center=True),
         p("Type",bold=True,center=True), p("vs SVM",bold=True,center=True)],
        [p("RA-MQTF",bold=True), p("0.834",bold=True,center=True),
         p("Quantum",center=True), p("+17.6%",center=True)],
        [p("Q-MoE Fire"), p("0.782",center=True), p("Quantum",center=True), p("+10.3%",center=True)],
        [p("SVM (RBF)"), p("0.709",center=True), p("Classical",center=True), p("—",center=True)],
        [p("XGBoost"), p("0.682",center=True), p("Classical",center=True), p("−3.8%",center=True)],
        [p("Logistic Regression"), p("0.673",center=True), p("Classical",center=True), p("−5.1%",center=True)],
        [p("Random Forest"), p("0.651",center=True), p("Classical",center=True), p("−8.2%",center=True)],
    ]
    story.append(make_table(cls_cmp_data,
        [2.5*inch, 0.9*inch, 1.3*inch, 1.0*inch],
        extra_style=[
            ("BACKGROUND", (0,1),(3,2), colors.HexColor("#D4EDDA")),
        ]))
    story.append(sp(4))

    story.append(body("<b>Progression of quantum models:</b>"))
    story += [
        bullet("Generation 1 (EDA): VQC F1=0.542 — underperforms classical (barren plateaus)"),
        bullet("Generation 2 (Cloud QML): QRC F1=0.724 — first quantum improvement over SVM (+2.1%)"),
        bullet("Generation 3 (Q-MoE): Q-MoE F1=0.782 — regime-aware experts (+10.3%)"),
        bullet("Generation 4 (RA-MQTF): RA-MQTF F1=0.834 — multi-task fusion (+17.6%)"),
        sp(6),
    ]

    # 3.4 Task 2
    story += [subsection("3.4  Task 2: Insurance Premiums and Fire Severity"), sp(2)]

    story.append(body(
        "<b>ZIP-level premium prediction — QLSTM</b>  "
        "(<i>02_quantum_baselines/03_Cloud_Quantum_Models</i>):"))
    story += [
        bullet("4-qubit variational circuits replace standard LSTM forget, input, cell, and "
               "output gates. StronglyEntanglingLayers ansatz, 3 layers per gate (144 quantum "
               "parameters total)."),
        bullet("2-step time sequences of ZIP-code insurance features predict next-year premium. "
               "Input includes the dataset's fire risk score and Q-MoE's county-level fire "
               "probability mapped to ZIP, directly connecting Task 1 output to Task 2."),
        bullet("<b>Result: R2 = 0.349 (temporal log-scale split), RMSE = 0.458</b>"),
        bullet("Substantially outperforms Classical LSTM (R2=−1.033) on the same temporal split, "
               "demonstrating that quantum gates capture temporal premium dynamics that standard "
               "recurrent cells miss. Temporal XGBoost (R2=0.828) remains the strongest temporal "
               "baseline; the premium task is inherently difficult with only 4 years of annual "
               "ZIP data."),
        sp(4),
    ]

    story.append(body(
        "<b>County-level fire severity — RA-MQTF</b>  "
        "(<i>04_proposed_algorithm/04_RegimeAware_MQTF</i>):"))
    story += [
        bullet("Same Q-MoE expert features with multi-task backbone predicting log(burned acres)."),
        bullet("<b>Result: R2 = 0.639, RMSE = 1.563</b> — best severity prediction in the project."),
        bullet("Outperforms Q-MoE single-task (R2=0.535), MQTF (R2=0.475), and QART (R2=0.279). "
               "Multi-task regularisation delivers +19.4% improvement over single-task training."),
        sp(4),
    ]

    story.append(body("<b>Supporting explorations:</b>"))
    story += [
        bullet("MQTF: multi-task learning provides +69% F1 and +588% R2 over single-task "
               "baselines, motivating RA-MQTF."),
        bullet("Q-CAST: 6 climate scenarios through the quantum pipeline; drought flips 35 "
               "county-months to fire-prone, with conformal 90% coverage intervals."),
        bullet("CQP (Conformal Quantum Prediction): ~90% empirical coverage across all regimes."),
        sp(6),
    ]

    # 3.5 Resource Requirements
    story += [subsection("3.5  Quantum Resource Requirements"), sp(2)]

    story.append(body(
        "All models run on PennyLane <i>default.qubit</i> (exact statevector simulator — "
        "analytic mode). AWS Braket integration for IonQ Aria, Rigetti Ankaa-3, and IQM "
        "Garnet is implemented in the Cloud Quantum Models notebook."))
    story.append(sp(4))

    res_data = [
        [p("Model",bold=True), p("Qubits",bold=True,center=True),
         p("Depth",bold=True,center=True), p("Train Params",bold=True,center=True),
         p("Total Shots",bold=True,center=True), p("Sim. Time",bold=True,center=True)],
        [p("Q-MoE (4 experts)"), p("4×4",center=True), p("40 each",center=True),
         p("0 fixed + GBM",center=True),
         p("Analytic\n(statevector)",center=True), p("~7 min",center=True)],
        [p("RA-MQTF (multi-task)"), p("4×4",center=True), p("40 each",center=True),
         p("0 fixed + NN (6,595)",center=True),
         p("Analytic\n(statevector)",center=True), p("~6.5 min",center=True)],
        [p("QLSTM (Task 2)"), p("4",center=True), p("12",center=True),
         p("144",center=True),
         p("~864,000\n(analytic grad)",center=True), p("~15 s",center=True)],
        [p("QRC (Task 1A baseline)"), p("8",center=True), p("60",center=True),
         p("0 fixed",center=True),
         p("Analytic\n(statevector)",center=True), p("~18 s",center=True)],
    ]
    story.append(make_table(res_data,
        [1.8*inch, 0.7*inch, 0.75*inch, 1.3*inch, 1.2*inch, 0.75*inch]))
    story.append(note(
        "Shot count note: Fixed reservoirs (Q-MoE, RA-MQTF, QRC) use exact statevector "
        "evaluation — equivalent to infinite shots. QLSTM shot estimate: "
        "144 params × 2 (parameter-shift) × 30 epochs × 200 training steps = 1,728,000 "
        "circuit evaluations in analytic mode."))
    story.append(sp(6))

    # ── Section 4: Results and Code Repository ───────────────────────────────
    story += [PageBreak(), section("Section 4: Results and Code Repository"), hr()]

    # Task coverage overview (moved from abstract)
    story += [subsection("4.0  Task Coverage Overview"), sp(2)]
    cov_data = [
        [p("Task",bold=True), p("Model",bold=True), p("Granularity",bold=True),
         p("Score",bold=True), p("Limitation",bold=True)],
        [p("1A: Wildfire classification"), p("RA-MQTF",bold=True),
         p("County → ZIP"), p("F1=0.834"),
         p("Not direct ZIP-level")],
        [p("1B: Evaluation"), p("15+ models benchmarked"),
         p("County + daily"), p("See §3.3"), p("—")],
        [p("2: Fire severity"), p("RA-MQTF",bold=True),
         p("County"), p("R2=0.639"), p("County, not ZIP")],
        [p("2: Insurance premiums"), p("QLSTM"),
         p("ZIP"), p("R2=0.349\n(temporal log-scale)"),
         p("XGBoost R2=0.828 stronger")],
        [p("Forward predictions"), p("Q-MoE + ensemble"),
         p("ZIP"), p("2,174 ZIPs scored"),
         p("County-to-ZIP mapping")],
    ]
    story.append(make_table(cov_data,
        [1.4*inch, 1.4*inch, 1.0*inch, 1.2*inch, 1.5*inch]))
    story.append(sp(6))

    story += [subsection("4.1  Results Summary"), sp(2)]
    res_sum_data = [
        [p("Task",bold=True), p("Model",bold=True), p("Granularity",bold=True),
         p("Score",bold=True), p("Classical Best",bold=True), p("Delta",bold=True)],
        [p("1A Classification"), p("RA-MQTF",bold=True), p("County"),
         p("F1=0.834",bold=True), p("SVM F1=0.709"), p("+17.6%",bold=True)],
        [p("1A Classification"), p("Q-MoE Fire"), p("County"),
         p("F1=0.782"), p("SVM F1=0.709"), p("+10.3%")],
        [p("2 Fire Severity"), p("RA-MQTF",bold=True), p("County"),
         p("R2=0.639",bold=True), p("—"), p("Best overall")],
        [p("2 Premium (temporal)"), p("QLSTM"), p("ZIP"),
         p("R2=0.349"), p("Classical LSTM R2=−1.033"), p("QLSTM wins")],
        [p("2 Premium (temporal)"), p("Temporal XGBoost"), p("ZIP"),
         p("R2=0.828"), p("—"), p("Best temporal classical")],
        [p("2 Premium (non-temporal)"), p("Random Forest"), p("ZIP"),
         p("R2=0.922"), p("—"), p("Non-temporal split")],
        [p("Forward Predictions"), p("Q-MoE + ensemble"), p("ZIP"),
         p("2,174 ZIPs"), p("—"), p("Per-ZIP risk map")],
    ]
    story.append(make_table(res_sum_data,
        [1.15*inch, 1.25*inch, 0.75*inch, 0.9*inch, 1.5*inch, 0.95*inch]))
    story.append(sp(6))

    story += [subsection("4.2  Rubric-Aligned Summary"), sp(2)]
    rubric_data = [
        [p("Criterion",bold=True), p("Evidence",bold=True), p("Location",bold=True)],
        [p("Task 1A: Wildfire risk"),
         p("RA-MQTF F1=0.834 (county → ZIP)"),
         p("04_proposed_algorithm/04_RegimeAware_MQTF\n05_predictions/01_ZipCode_2026_Predictions")],
        [p("Task 1B: Evaluation"),
         p("15+ models benchmarked, ablations, per-regime validation, SHAP"),
         p("Sections 3.2–3.3; all baseline notebooks")],
        [p("Task 2: Premium/severity"),
         p("QLSTM R2=0.349 premiums (beats Classical LSTM R2=−1.033)\nRA-MQTF R2=0.639 severity"),
         p("02_quantum_baselines/03_Cloud_Quantum_Models\n04_proposed_algorithm/04_RegimeAware_MQTF")],
        [p("Novel algorithm"),
         p("Q-MoE (regime experts) + RA-MQTF (multi-task fusion)"),
         p("04_proposed_algorithm/")],
        [p("Envisioned algorithm"),
         p("ZIP-STR-QMoE: all validated ingredients at ZIP granularity"),
         p("Section 5")],
        [p("Quantum hardware path"),
         p("4-qubit fixed reservoirs; AWS Braket for IonQ/Rigetti/IQM"),
         p("02_quantum_baselines/03_Cloud_Quantum_Models")],
        [p("Known limitation"),
         p("County-level wildfire models mapped to ZIPs. Proposed algorithm addresses this."),
         p("Sections 2, 5")],
    ]
    story.append(make_table(rubric_data,
        [1.5*inch, 2.5*inch, 2.5*inch]))
    story.append(sp(6))

    story += [subsection("4.3  Notebooks (17 total, organised in 5 folders)"), sp(2)]
    nb_data = [
        [p("Folder",bold=True), p("Notebook",bold=True), p("Purpose",bold=True)],
        [p("01_eda_and_visualization/"), p("01 EDA and Baselines"),
         p("Classical + quantum baselines, Task 1A")],
        [p(""), p("02 Geospatial Visualization"),
         p("Risk maps + quantum resource analysis")],
        [p(""), p("03 Clustering UMAP HDBSCAN"),
         p("ZIP-code risk clustering into 5 tiers")],
        [p(""), p("04 FullHistory Wildfire Analysis"),
         p("40-year wildfire trend analysis")],
        [p("02_quantum_baselines/"), p("01 Task2 Insurance QML"),
         p("Classical + quantum baselines, Task 2")],
        [p(""), p("02 Improved Quantum Models"),
         p("PCA, data re-uploading, hybrid ensembles")],
        [p(""), p("03 Cloud Quantum Models"),
         p("QRC, QLSTM, Trainable Kernel, Transfer Learning")],
        [p(""), p("04 Advanced Quantum Architectures"),
         p("QCNN, PQK, QGAN, Quanvolution, Dressed Circuit")],
        [p("03_novel_explorations/"), p("01 RAM-QRSRE"),
         p("Regime-adaptive multi-scale QML")],
        [p(""), p("02 Quantum Graph Reservoir"),
         p("Spatial graph + quantum reservoir")],
        [p(""), p("03 Conformal Quantum Prediction"),
         p("Uncertainty quantification with coverage guarantees")],
        [p(""), p("04 Multi-Task Quantum Temporal Fusion"),
         p("Joint fire + insurance prediction")],
        [p("04_proposed_algorithm/"), p("01 Q-MoE Fire  ★"),
         p("Regime-diverse quantum experts — F1=0.782")],
        [p(""), p("02 QART"),
         p("Quantum Adaptive Reservoir Transformer")],
        [p(""), p("03 Q-CAST"),
         p("Scenario analysis + decision support")],
        [p(""), p("04 RA-MQTF  ★"),
         p("Multi-task fusion — F1=0.834, R2=0.639")],
        [p("05_predictions/"), p("01 ZipCode 2026 Predictions"),
         p("Final per-ZIP wildfire risk predictions for 2,174 ZIPs")],
    ]
    story.append(make_table(nb_data,
        [1.85*inch, 2.15*inch, 2.5*inch]))
    story.append(note("★ Primary competition entries"))
    story.append(sp(6))

    story += [subsection("4.4  Code Repository"), sp(4)]
    story.append(Paragraph(
        '<a href="https://github.com/Priti0427/Quantum-Sustainability-Challenge" color="#2E5B9A">'
        'https://github.com/Priti0427/Quantum-Sustainability-Challenge</a>',
        S["repo"]))
    story.append(sp(2))
    story.append(body(
        "The repository is <b>public</b>. All 17 notebooks include pre-executed cell outputs "
        "for full reproducibility without re-running. Results are saved as JSON and CSV files "
        "in the <i>results/</i> directory. See <i>README.md</i> for setup instructions and "
        "pipeline documentation."))
    story.append(sp(6))

    # ── Section 5: Envisioned Algorithm ──────────────────────────────────────
    story += [PageBreak(), section("Section 5: Envisioned Algorithm — ZIP-STR-QMoE"), hr()]

    story.append(subsection(
        "ZIP-level Spatio-Temporal Regime-aware Quantum Mixture-of-Experts"))
    story.append(sp(2))

    story.append(body(
        "<b>Quantum community impact:</b> ZIP-STR-QMoE establishes a reusable pattern for "
        "hybrid quantum ML in any geospatial prediction problem with spatial heterogeneity, "
        "multi-task structure, and temporal dynamics — flood risk, crop yield, infrastructure "
        "resilience, and beyond. The three core ingredients (regime-diverse fixed reservoirs, "
        "multi-task fusion, QLSTM temporal head) are each independently validated and "
        "composable, enabling the community to adopt them selectively without re-implementing "
        "the full system."))
    story.append(sp(4))

    story.append(body(
        "<b>The gap this addresses:</b> Our validated models (Q-MoE, RA-MQTF) produce strong "
        "wildfire predictions at county granularity, but the challenge asks for ZIP-code-level "
        "predictions. Currently, we bridge this gap by mapping county fire probabilities to "
        "ZIPs — losing spatial precision. ZIP-STR-QMoE performs prediction directly at ZIP level."))
    story.append(sp(6))

    story += [subsubsection("Architecture"), sp(2)]

    zip_arch = [
        ("<b>1. ZIP-level regime routing</b> (validated by Q-MoE at county scale): Instead of "
         "clustering 58 counties into 4 regimes, cluster ~2,000 ZIP codes into 6–8 finer "
         "regimes using ZIP-level insurance features, CalFire risk scores, and local weather. "
         "Each ZIP-month is routed to the matching quantum expert. Q-MoE proves regime routing "
         "improves all climate zones (4/4)."),
        ("<b>2. Topology-diverse quantum reservoir experts</b> (validated by Q-MoE): Same "
         "4-qubit fixed reservoirs (ring, ladder, star, full). No architectural change — only "
         "retrain on ZIP-level features. Fixed reservoirs avoid barren plateaus entirely."),
        ("<b>3. Multi-task shared backbone</b> (validated by RA-MQTF): Shared neural network "
         "jointly predicts wildfire risk, severity, and premiums. RA-MQTF proves multi-task "
         "training improves fire classification +6.6% and severity +19.4%."),
        ("<b>4. QLSTM premium head</b> (validated by QLSTM): Replace the simple regression "
         "head with a QLSTM temporal module on 2-step ZIP-level insurance sequences. QLSTM "
         "beats Classical LSTM by R2=0.349 vs −1.033 at ZIP granularity."),
        ("<b>5. Conformal uncertainty calibration</b> (validated by CQP): Wrap all predictions "
         "with adaptive conformal intervals providing 90% coverage guarantees — empirically "
         "validated in the CQP notebook."),
    ]
    for txt in zip_arch:
        story.append(bullet(txt))
    story.append(sp(6))

    story += [subsubsection("What Changes vs. Current System"), sp(2)]
    diff_data = [
        [p("Component",bold=True), p("Current — county",bold=True),
         p("ZIP-STR-QMoE — ZIP",bold=True)],
        [p("Granularity"), p("58 counties"), p("~2,000 ZIP codes")],
        [p("Regime clustering"), p("4 county regimes"), p("6–8 ZIP regimes")],
        [p("Input features"),
         p("Weather + fire history"),
         p("Weather + fire history + insurance + CalFire FHSZ")],
        [p("Premium prediction"), p("Mapped from county"), p("Direct ZIP-level QLSTM")],
        [p("Quantum circuits"),
         p("4-qubit reservoirs (fixed)"), p("Same 4-qubit reservoirs (unchanged)")],
    ]
    story.append(make_table(diff_data,
        [1.6*inch, 2.3*inch, 2.6*inch]))
    story.append(sp(6))

    story += [subsubsection("Why We Believe This Will Work"), sp(2)]
    story.append(body(
        "Every individual component is already validated with reproducible metrics. The main "
        "engineering challenge is the data pipeline — joining wildfire county data to ZIP-level "
        "insurance records requires a county-to-ZIP mapping layer during training. This is an "
        "implementation task, not an algorithmic uncertainty."))
    story.append(sp(4))

    story += [subsubsection("Quantum Hardware Requirements"), sp(2)]
    hw_data = [
        [p("Metric",bold=True), p("Value",bold=True)],
        [p("Logical qubits"), p("4 per expert × 4 experts = 16 total")],
        [p("Circuit depth"), p("40 layers per reservoir (NISQ-compatible with error mitigation)")],
        [p("Trainable quantum params"), p("0 (fixed reservoirs) + 144 (QLSTM premium head)")],
        [p("Compatible hardware"),
         p("IonQ Aria (25-qubit), IBM Eagle (127-qubit), Rigetti Ankaa-3")],
        [p("Estimated simulation time"), p("~15 min for full ~2,000 ZIP encoding + training")],
        [p("Near-term path"),
         p("Fixed reservoirs are noise-resilient; QLSTM head is the only component "
           "requiring quantum parameter optimisation")],
    ]
    t = Table(hw_data, colWidths=[2.0*inch, 4.5*inch],
              style=TableStyle([
                  ("GRID",        (0,0), (-1,-1), 0.4, MID_GRAY),
                  ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
                  ("FONTSIZE",    (0,0), (-1,-1), 9),
                  ("ROWBACKGROUNDS",(0,0),(-1,-1), [LIGHT_BLUE, colors.white]),
                  ("TOPPADDING",  (0,0), (-1,-1), 4),
                  ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                  ("LEFTPADDING", (0,0), (-1,-1), 6),
                  ("RIGHTPADDING",(0,0), (-1,-1), 6),
                  ("VALIGN",      (0,0), (-1,-1), "TOP"),
              ]))
    story.append(t)
    story.append(sp(8))

    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_GOLD,
                            spaceAfter=6, spaceBefore=4))
    story.append(body(
        "<i>Submitted for the Deloitte Quantum Sustainability Challenge 2026 — Phase 1. "
        "All code, data, and results are available at "
        "https://github.com/Priti0427/Quantum-Sustainability-Challenge</i>"))

    doc.build(story)
    print(f"PDF written to {outfile}")

if __name__ == "__main__":
    build_pdf("SUBMISSION.pdf")
