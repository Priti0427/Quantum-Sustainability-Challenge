# Deloitte Quantum Sustainability Challenge 2026

**Team: GenQ**
*Hybrid Quantum-Classical Wildfire Risk Prediction & Insurance Premium Estimation*

---

## 1. Team Overview

- Priti Sagar (pp693@drexel.edu, Drexel University)
- Gautam Mahajan (gautam.mahajan@uconn.edu, University of Connecticut – Storrs)

---

## 2. Abstract

We present a hybrid quantum-classical system for California wildfire risk prediction and insurance premium estimation, built in three validated layers.

**Layer 1: Q-MoE Fire (regime-diverse quantum experts):** California's 58 counties cluster into 4 climate regimes with fundamentally different fire dynamics. Q-MoE routes each county-month through 4 quantum reservoir experts, each using a different entanglement topology (ring, ladder, star, full). The concatenated 44-dimensional feature vector feeds a GradientBoosting head. Q-MoE achieves F1=0.782 and AUC=0.934, outperforming SVM (F1=0.709) by 10.3%. Per-regime ablation confirms improvements in all 4 climate zones.

**Layer 2: RA-MQTF (multi-task fusion):** RA-MQTF adds a shared multi-task neural backbone on top of Q-MoE's expert features, jointly predicting fire occurrence, severity, and insurance premiums. Multi-task training improves fire classification to F1=0.834 (+6.6% over Q-MoE) and severity to R2=0.639 (+19.4%). The premium task via county-to-ZIP mapping confirms that the county granularity is the binding constraint, exactly the gap ZIP-STR-QMoE (Section 5) addresses directly.

**Layer 3: QLSTM (temporal premium model):** 4-qubit variational LSTM gates achieve R2=0.349 on temporal log-scale splits, substantially outperforming Classical LSTM (R2=−1.033). With only 4 years of annual ZIP data available, QLSTM's advantage demonstrates that quantum gates capture temporal premium dynamics that standard recurrent cells miss.

**Main limitation:** Q-MoE and RA-MQTF operate at county granularity; ZIP-level predictions use a county-to-ZIP geographic mapping. The proposed ZIP-STR-QMoE algorithm (Section 5) addresses this directly.

**Forward-looking predictions:** Per-ZIP wildfire risk scores for 2,174 California locations, combining Q-MoE county fire probabilities with domain-driven indicators.

**Proposed algorithm:** ZIP-STR-QMoE (Section 5) extends these validated ingredients to direct ZIP-level prediction, addressing the challenge's core granularity requirement.

---

## 3. Detailed Algorithm Description

### 3.1 Data Pipeline

**a. Challenge-provided datasets**

**b. Additional data used: None.** All analysis uses only the challenge-provided files:
- `wildfire_county_monthly.csv`: monthly county-level wildfire and weather records (2008–2020)
- `wildfire_weather_daily.csv`: daily weather and fire occurrence records, used for temporal trend analysis and 2026 forward projections
- `insurance_2018_2019.XLS` and `insurance_2020_2021.XLS`: ZIP-level homeowners insurance premiums and risk scores; `geojson-counties-fips.json`: county boundaries for geospatial visualization

No external APIs, third-party datasets, or real-time data feeds were used.

**Feature engineering:** Temperature range, wind–temperature ratio, lagged precipitation and wind speed, PCA-reduced quantum encodings (4 components), county adjacency graph features, temporal lag features (lag_acres, lag_temp), and domain-driven indicators (Fire Weather Index proxy, drought severity, hydroclimate whiplash signals).

**Temporal split:** Train 2008–2018 / test 2019–2020 (county wildfire); train 2018–2020 / test 2021 (ZIP insurance).

*Note: the provided `wildfire_county_monthly.csv` ends at 2020-12. Q-MoE and RA-MQTF use 2008–2018 training to preserve a held-out 2019–2020 test set. For the challenge's 2018–2021 window, the insurance dataset is used in full for Task 2, and the daily weather data (extending to 2025) supplements county wildfire data for trend analysis and 2026 forward projections.*

### 3.2 Task 1A: Wildfire Classification

**Primary model: Q-MoE Fire - Quantum Mixture-of-Experts** (`04_proposed_algorithm/01_QMoE_Fire`)

California spans 16 climate zones with fundamentally different fire dynamics. A single quantum model forced to learn all regimes simultaneously must average across them. Q-MoE instead routes each county-month to specialised quantum experts matched to its climate regime.

**Architecture**

1. **Regime discovery:** KMeans clustering on county-level geography and weather produces 4 natural climate regimes:
   - Regime A (24 counties): Moderate coastal/central California - 4.9% fire rate
   - Regime B (13 counties): Hot, fire-prone inland (Fresno, Kern, L.A.) - 39% fire rate
   - Regime C (17 counties): Cool, wet northern (Humboldt, Shasta, Butte) - 20% fire rate
   - Regime D (4 counties): Extreme desert (Imperial, Inyo, Riverside, San Bernardino) - 24% fire rate
2. **Four quantum reservoir experts:** Each expert uses 4 qubits and 10 reservoir layers with fixed (non-trainable) random weights and genuinely different entanglement topologies — ring (circular CNOT chain), ladder (alternating even/odd pairs), star (hub-and-spoke), and full (all-to-all). Each expert produces 11 observables (4 PauliZ + 4 PauliX + 3 nearest-neighbour ZZ correlations).
3. **Concat gating:** All 44 expert features (4×11) are concatenated and fed to a GradientBoosting head that learns which expert features matter for each sample. This outperformed explicit hard gating and soft gating.

**Result: F1 = 0.782, AUC = 0.934**

**Multi-task improvement: RA-MQTF** (`04_proposed_algorithm/04_RegimeAware_MQTF`): Adding a shared multi-task backbone on top of Q-MoE expert features further improves fire classification to **F1 = 0.834** (+6.6%) and fire severity to **R2 = 0.639** (+19.4%). Joint training acts as regularisation.

**Per-Regime Validation:** Q-MoE improves all 4 climate zones

| Regime | N test | Fire Rate | MoE F1 | Baseline F1 | F1 Lift |
|--------|--------|-----------|--------|-------------|---------|
| A. coastal | 570 | 7.4% | 0.693 | 0.582 | +0.111 |
| B. inland hot | 308 | 41.9% | 0.833 | 0.736 | +0.098 |
| C. northern wet | 402 | 22.9% | 0.739 | 0.680 | +0.059 |
| D. desert | 94 | 24.5% | 0.818 | 0.708 | +0.110 |

**All Quantum Classifiers Explored** (identical test splits)

| Model | F1 | Qubits | Notebook |
|-------|----|--------|----------|
| **RA-MQTF (multi-task)** | **0.834** | 4×4 | `04_proposed_algorithm/04_RegimeAware_MQTF` |
| Q-MoE Concat + GBM | 0.782 | 4×4 | `04_proposed_algorithm/01_QMoE_Fire` |
| QART (cross-attention) | 0.755 | 4 | `04_proposed_algorithm/02_QART` |
| QRC + LogReg | 0.724 | 8 | `02_quantum_baselines/03_Cloud_Quantum_Models` |
| MQTF (multi-task) | 0.693 | 4 | `03_novel_explorations/04_MultiTask_QTF` |
| QGR (graph) | 0.650 | 4 | `03_novel_explorations/02_Quantum_Graph_Reservoir` |
| Dressed Quantum Circuit | 0.637 | 4 | `02_quantum_baselines/04_Advanced_Architectures` |
| VQC (variational) | 0.542 | 6 | `01_eda_and_visualization/01_EDA_and_Baselines` |

### 3.3 Task 1B: Evaluation

**Advantages of our quantum approach:**
- **No barren plateaus:** Fixed quantum reservoirs (QRC, Q-MoE) have zero trainable quantum parameters, completely avoiding the barren plateau problem that undermines variational methods (VQC achieved only F1=0.542).
- **Regime specialisation:** Different entanglement topologies create diverse feature maps capturing different fire dynamics — validated by per-regime improvements in all 4 climate zones.
- **NISQ-ready:** 4-qubit reservoirs at depth 40 are deployable on current hardware (IonQ Aria 25-qubit, Rigetti Ankaa-3). AWS Braket integration is implemented in the Cloud Quantum Models notebook.
- **Interpretable:** Classical GBM and LogReg heads provide feature importance; SHAP analysis identifies key wildfire drivers.

**Limitations:**
- **Quantum encoding overhead:** Encoding 8,584 samples through 4 experts takes ~7 minutes on a simulator with no parallelisation advantage over classical GBM.
- **Indirect quantum benefit:** The quantum reservoir is a non-linear feature extractor; classification is performed by a classical head. Benefit comes from the feature space, not quantum speedup.
- **County granularity:** Q-MoE and RA-MQTF operate at county level; ZIP predictions require a mapping step that reduces spatial precision.

**Classical comparison** (identical data, identical splits):

| Model | F1 | Type | vs SVM |
|-------|-----|------|--------|
| **RA-MQTF** | **0.834** | Quantum | +17.6% |
| Q-MoE Fire | 0.782 | Quantum | +10.3% |
| SVM (RBF) | 0.709 | Classical | — |
| XGBoost | 0.682 | Classical | −3.8% |
| Logistic Regression | 0.673 | Classical | −5.1% |
| Random Forest | 0.651 | Classical | −8.2% |

**Progression of quantum models:**
- Generation 1 (EDA): VQC F1=0.542 — underperforms classical (barren plateaus)
- Generation 2 (Cloud QML): QRC F1=0.724 — first quantum improvement over SVM (+2.1%)
- Generation 3 (Q-MoE): Q-MoE F1=0.782 — regime-aware experts (+10.3%)
- Generation 4 (RA-MQTF): RA-MQTF F1=0.834 — multi-task fusion (+17.6%)

### 3.4 Task 2: Insurance Premiums and Fire Severity

**ZIP-level premium prediction: QLSTM** (`02_quantum_baselines/03_Cloud_Quantum_Models`):
- 4-qubit variational circuits replace standard LSTM forget, input, cell, and output gates. StronglyEntanglingLayers ansatz, 3 layers per gate (144 quantum parameters total).
- 2-step time sequences of ZIP-code insurance features predict next-year premium. Input includes the dataset's fire risk score and Q-MoE's county-level fire probability mapped to ZIP, directly connecting Task 1 output to Task 2.
- **Result: R2 = 0.349 (temporal log-scale split), RMSE = 0.458**
- Substantially outperforms Classical LSTM (R2=−1.033) on the same temporal split, demonstrating that quantum gates capture temporal premium dynamics that standard recurrent cells miss. Temporal XGBoost (R2=0.828) remains the strongest temporal baseline; the premium task is inherently difficult with only 4 years of annual ZIP data.

**County-level fire severity: RA-MQTF** (`04_proposed_algorithm/04_RegimeAware_MQTF`):
- Same Q-MoE expert features with multi-task backbone predicting log(burned acres).
- **Result: R2 = 0.639, RMSE = 1.563** — best severity prediction in the project.
- Outperforms Q-MoE single-task (R2=0.535), MQTF (R2=0.475), and QART (R2=0.279). Multi-task regularisation delivers +19.4% improvement over single-task training.

**Supporting explorations:**
- **MQTF:** multi-task learning provides +69% F1 and +588% R2 over single-task baselines, motivating RA-MQTF.
- **Q-CAST:** 6 climate scenarios through the quantum pipeline; drought flips 35 county-months to fire-prone, with conformal 90% coverage intervals.
- **CQP (Conformal Quantum Prediction):** ~90% empirical coverage across all regimes.

### 3.5 Quantum Resource Requirements

All models run on PennyLane `default.qubit` (exact statevector simulator - analytic mode). AWS Braket integration for IonQ Aria, Rigetti Ankaa-3, and IQM Garnet is implemented in the Cloud Quantum Models notebook.

| Model | Qubits | Depth | Train Params | Total Shots | Sim. Time |
|-------|--------|-------|--------------|-------------|-----------|
| Q-MoE (4 experts) | 4×4 | 40 each | 0 fixed + GBM | Analytic (statevector) | ~7 min |
| RA-MQTF (multi-task) | 4×4 | 40 each | 0 fixed + NN (6,595) | Analytic (statevector) | ~6.5 min |
| QLSTM (Task 2) | 4 | 12 | 144 | ~864,000 (analytic grad) | ~15 s |
| QRC (Task 1A baseline) | 8 | 60 | 0 fixed | Analytic (statevector) | ~18 s |

*Shot count note: Fixed reservoirs (Q-MoE, RA-MQTF, QRC) use exact statevector evaluation — equivalent to infinite shots. QLSTM shot estimate: 144 params × 2 (parameter-shift) × 30 epochs × 200 training steps = 1,728,000 circuit evaluations in analytic mode.*

---

## Section 4: Results and Code Repository

### 4.0 Task Coverage Overview

| Task | Model | Granularity | Score | Limitation |
|------|-------|-------------|-------|------------|
| 1A: Wildfire classification | **RA-MQTF** | County → ZIP | F1=0.834 | Not direct ZIP-level |
| 1B: Evaluation | 15+ models benchmarked | County + daily | See §3.3 | — |
| 2: Fire severity | **RA-MQTF** | County | R2=0.639 | County, not ZIP |
| 2: Insurance premiums | QLSTM | ZIP | R2=0.349 (temporal log-scale) | XGBoost R2=0.828 stronger |
| Forward predictions | Q-MoE + ensemble | ZIP | 2,174 ZIPs scored | County-to-ZIP mapping |

### 4.1 Results Summary

| Task | Model | Granularity | Score | Classical Best | Delta |
|------|-------|-------------|-------|----------------|-------|
| 1A Classification | **RA-MQTF** | County | **F1=0.834** | SVM F1=0.709 | +17.6% |
| 1A Classification | Q-MoE Fire | County | F1=0.782 | SVM F1=0.709 | +10.3% |
| 2 Fire Severity | **RA-MQTF** | County | **R2=0.639** | — | Best overall |
| 2 Premium (temporal) | QLSTM | ZIP | R2=0.349 | Classical LSTM R2=−1.033 | QLSTM wins |
| 2 Premium (temporal) | Temporal XGBoost | ZIP | R2=0.828 | — | Best temporal classical |
| 2 Premium (non-temporal) | Random Forest | ZIP | R2=0.922 | — | Non-temporal split |
| Forward Predictions | Q-MoE + ensemble | ZIP | 2,174 ZIPs | — | Per-ZIP risk map |

### 4.2 Rubric-Aligned Summary

| Criterion | Evidence | Location |
|-----------|----------|----------|
| Task 1A: Wildfire risk | RA-MQTF F1=0.834 (county → ZIP) | `04_proposed_algorithm/04_RegimeAware_MQTF`, `05_predictions/01_ZipCode_2026_Predictions` |
| Task 1B: Evaluation | 15+ models benchmarked, ablations, per-regime validation, SHAP | Sections 3.2–3.3; all baseline notebooks |
| Task 2: Premium/severity | QLSTM R2=0.349 premiums (beats Classical LSTM R2=−1.033); RA-MQTF R2=0.639 severity | `02_quantum_baselines/03_Cloud_Quantum_Models`, `04_proposed_algorithm/04_RegimeAware_MQTF` |
| Novel algorithm | Q-MoE (regime experts) + RA-MQTF (multi-task fusion) | `04_proposed_algorithm/` |
| Envisioned algorithm | ZIP-STR-QMoE: all validated ingredients at ZIP granularity | Section 5 |
| Quantum hardware path | 4-qubit fixed reservoirs; AWS Braket for IonQ/Rigetti/IQM | `02_quantum_baselines/03_Cloud_Quantum_Models` |
| Known limitation | County-level wildfire models mapped to ZIPs. Proposed algorithm addresses this. | Sections 2, 5 |

### 4.3 Notebooks (17 total, organised in 5 folders)

| Folder | Notebook | Purpose |
|--------|----------|---------|
| `01_eda_and_visualization/` | 01 EDA and Baselines | Classical + quantum baselines, Task 1A |
| | 02 Geospatial Visualization | Risk maps + quantum resource analysis |
| | 03 Clustering UMAP HDBSCAN | ZIP-code risk clustering into 5 tiers |
| | 04 FullHistory Wildfire Analysis | 40-year wildfire trend analysis |
| `02_quantum_baselines/` | 01 Task2 Insurance QML | Classical + quantum baselines, Task 2 |
| | 02 Improved Quantum Models | PCA, data re-uploading, hybrid ensembles |
| | 03 Cloud Quantum Models | QRC, QLSTM, Trainable Kernel, Transfer Learning |
| | 04 Advanced Quantum Architectures | QCNN, PQK, QGAN, Quanvolution, Dressed Circuit |
| `03_novel_explorations/` | 01 RAM-QRSRE | Regime-adaptive multi-scale QML |
| | 02 Quantum Graph Reservoir | Spatial graph + quantum reservoir |
| | 03 Conformal Quantum Prediction | Uncertainty quantification with coverage guarantees |
| | 04 Multi-Task Quantum Temporal Fusion | Joint fire + insurance prediction |
| **`04_proposed_algorithm/`** | **01 Q-MoE Fire ★** | Regime-diverse quantum experts — F1=0.782 |
| | 02 QART | Quantum Adaptive Reservoir Transformer |
| | 03 Q-CAST | Scenario analysis + decision support |
| | **04 RA-MQTF ★** | Multi-task fusion — F1=0.834, R2=0.639 |
| `05_predictions/` | 01 ZipCode 2026 Predictions | Final per-ZIP wildfire risk predictions for 2,174 ZIPs |

★ *Primary competition entries*

### 4.4 Code Repository

**https://github.com/Priti0427/Quantum-Sustainability-Challenge**

The repository is public. All 17 notebooks include pre-executed cell outputs for full reproducibility without re-running. Results are saved as JSON and CSV files in the `results/` directory. See README.md for setup instructions and pipeline documentation.

---

## Section 5: Envisioned Algorithm — ZIP-STR-QMoE

### ZIP-level Spatio-Temporal Regime-aware Quantum Mixture-of-Experts

**Quantum community impact:** ZIP-STR-QMoE establishes a reusable pattern for hybrid quantum ML in any geospatial prediction problem with spatial heterogeneity, multi-task structure, and temporal dynamics — flood risk, crop yield, infrastructure resilience, and beyond. The three core ingredients (regime-diverse fixed reservoirs, multi-task fusion, QLSTM temporal head) are each independently validated and composable, enabling the community to adopt them selectively without re-implementing the full system.

**The gap this addresses:** Our validated models (Q-MoE, RA-MQTF) produce strong wildfire predictions at county granularity, but the challenge asks for ZIP-code-level predictions. Currently, we bridge this gap by mapping county fire probabilities to ZIPs — losing spatial precision. ZIP-STR-QMoE performs prediction directly at ZIP level.

### Architecture

1. **ZIP-level regime routing** (validated by Q-MoE at county scale): Instead of clustering 58 counties into 4 regimes, cluster ~2,000 ZIP codes into 6–8 finer regimes using ZIP-level insurance features, CalFire risk scores, and local weather. Each ZIP-month is routed to the matching quantum expert. Q-MoE proves regime routing improves all climate zones (4/4).
2. **Topology-diverse quantum reservoir experts** (validated by Q-MoE): Same 4-qubit fixed reservoirs (ring, ladder, star, full). No architectural change — only retrain on ZIP-level features. Fixed reservoirs avoid barren plateaus entirely.
3. **Multi-task shared backbone** (validated by RA-MQTF): Shared neural network jointly predicts wildfire risk, severity, and premiums. RA-MQTF proves multi-task training improves fire classification +6.6% and severity +19.4%.
4. **QLSTM premium head** (validated by QLSTM): Replace the simple regression head with a QLSTM temporal module on 2-step ZIP-level insurance sequences. QLSTM beats Classical LSTM by R2=0.349 vs −1.033 at ZIP granularity.
5. **Conformal uncertainty calibration** (validated by CQP): Wrap all predictions with adaptive conformal intervals providing 90% coverage guarantees — empirically validated in the CQP notebook.

### What Changes vs. Current System

| Component | Current — county | ZIP-STR-QMoE — ZIP |
|-----------|------------------|---------------------|
| Granularity | 58 counties | ~2,000 ZIP codes |
| Regime clustering | 4 county regimes | 6–8 ZIP regimes |
| Input features | Weather + fire history | Weather + fire history + insurance + CalFire FHSZ |
| Premium prediction | Mapped from county | Direct ZIP-level QLSTM |
| Quantum circuits | 4-qubit reservoirs (fixed) | Same 4-qubit reservoirs (unchanged) |

### Why We Believe This Will Work

Every individual component is already validated with reproducible metrics. The main engineering challenge is the data pipeline: joining wildfire county data to ZIP-level insurance records requires a county-to-ZIP mapping layer during training. This is an implementation task, not an algorithmic uncertainty.

### Quantum Hardware Requirements

| Metric | Value |
|--------|-------|
| Logical qubits | 4 per expert × 4 experts = 16 total |
| Circuit depth | 40 layers per reservoir (NISQ-compatible with error mitigation) |
| Trainable quantum params | 0 (fixed reservoirs) + 144 (QLSTM premium head) |
| Compatible hardware | IonQ Aria (25-qubit), IBM Eagle (127-qubit), Rigetti Ankaa-3 |
| Estimated simulation time | ~15 min for full ~2,000 ZIP encoding + training |
| Near-term path | Fixed reservoirs are noise-resilient; QLSTM head is the only component requiring quantum parameter optimization |

---

*Submitted for the Deloitte Quantum Sustainability Challenge 2026. All code, data, and results are available at https://github.com/Priti0427/Quantum-Sustainability-Challenge*
