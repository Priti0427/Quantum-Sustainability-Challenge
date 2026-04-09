# Deloitte Quantum Sustainability Challenge - Submission

---

## Section 1: Team Overview

**Team Name**: GenQ

| Member | University | Email |
|--------|-----------|-------|
| Priti Sagar | Drexel University | pp693@drexel.edu |
| Gautam Mahajan | University of Connecticut - Storrs | gautam.mahajan@uconn.edu |

---

## Section 2: Abstract 

We present a hybrid quantum-classical system for California wildfire risk prediction and insurance premium estimation, built in three validated layers.

**Layer 1 - Q-MoE Fire (regime-diverse quantum experts)**: California's 58 counties cluster into 4 climate regimes with fundamentally different fire dynamics. Q-MoE routes each county-month through 4 quantum reservoir experts, each using a different entanglement topology (ring, ladder, star, full). The concatenated 44-dimensional feature vector feeds a classical GradientBoosting head. Q-MoE achieves F1=0.782 and AUC=0.934 for wildfire classification, outperforming SVM (F1=0.709) by 10.3%. Per-regime ablation confirms improvements in all 4 climate zones.

**Layer 2 - RA-MQTF (multi-task fusion)**: Building on Q-MoE's expert features, RA-MQTF adds a shared multi-task neural backbone that jointly predicts fire occurrence, fire severity, and insurance premiums. Multi-task training improves fire classification to F1=0.834 (+6.6% over Q-MoE) and severity to R2=0.639 (+19.4% over Q-MoE). Premium prediction remains weak (R2=-1.276), confirming that annual ZIP-level premiums mapped to monthly county data lack sufficient signal for this task.

**Layer 3 - QLSTM (temporal premium model)**: For ZIP-level premium time-series, QLSTM (4-qubit variational LSTM gates) achieves R2=0.349 on temporal log-scale splits, substantially outperforming Classical LSTM (R2=-1.033). Temporal XGBoost (R2=0.828) remains the strongest temporal baseline. The premium task is inherently difficult with only 4 years of annual ZIP data; QLSTM's advantage over classical LSTM demonstrates that quantum gates capture temporal structure that standard recurrent cells miss.

**Main limitation**: Our strongest wildfire models operate at county granularity and are mapped to ZIP codes via geographic lookup. The challenge asks for direct ZIP-level prediction. This county-to-ZIP transfer is our main approximation, and our proposed algorithm (Section 5) addresses exactly this gap.

**Forward-Looking Predictions**: Per-ZIP wildfire risk scores for 2,174 California locations, combining Q-MoE county fire probabilities with domain-driven indicators.

**Task coverage at a glance**:

| Task | Model | Granularity | Score | Limitation |
|------|-------|-------------|-------|------------|
| 1A: Wildfire classification | RA-MQTF | County (mapped to ZIP) | F1=0.834 | Not direct ZIP-level |
| 1B: Evaluation | 15+ models benchmarked | County + daily | See Section 3.3 | - |
| 2: Fire severity | RA-MQTF | County | R2=0.639 | County, not ZIP |
| 2: Insurance premiums | QLSTM | ZIP | R2=0.349 (log-scale temporal) | XGBoost R2=0.828 stronger |
| Forward predictions | Q-MoE + ensemble | ZIP | 2,174 ZIPs scored | County-to-ZIP mapping |

**Proposed algorithm**: ZIP-STR-QMoE (Section 5) extends these validated ingredients to direct ZIP-level prediction, addressing the challenge's core granularity requirement.

---

## Section 3: Detailed Algorithm Description

### 3.1 Data Pipeline

**Challenge-provided datasets**:
- **Wildfire data**: California wildfire records including fire name, location, county, burned acreage (GIS_ACRES), organized at monthly county-level granularity (10,989 rows, 58 counties, 2008-2020)
- **Insurance data**: California Department of Insurance homeowners insurance filings - `insurance_2018_2019.XLS` and `insurance_2020_2021.XLS` - containing ZIP-level premiums, exposures, losses, and risk scores (~2,000 ZIPs per year)

**Additional data used**: None. All analysis uses only the five challenge-provided files:
- `wildfire_county_monthly.csv` — monthly county-level wildfire and weather records (2008–2020)
- `wildfire_weather_daily.csv` — daily weather and fire occurrence records (used for temporal analysis and 2026 projections)
- `insurance_2018_2019.XLS` — ZIP-level homeowners insurance premiums and risk scores
- `insurance_2020_2021.XLS` — ZIP-level homeowners insurance premiums and risk scores
- `geojson-counties-fips.json` — California county boundaries for geospatial visualisation

No external APIs, third-party datasets, or real-time data feeds were used.

**Feature engineering**: Temperature range, wind-temperature ratio, lagged precipitation and wind speed, PCA-reduced quantum encodings (4 components), county adjacency graph features, temporal lag features (lag_acres, lag_temp), and domain-driven indicators (Fire Weather Index proxy, drought severity, hydroclimate whiplash signals).

**Temporal split**: Train on 2008-2018, test on 2019-2020 (county-level wildfire); train on 2018-2020, test on 2021 (ZIP-level insurance). Note: the provided `wildfire_county_monthly.csv` ends at 2020-12; Q-MoE and RA-MQTF use 2008–2018 training to preserve a held-out 2019–2020 test set. For the challenge's 2018–2021 window, the insurance dataset (2018–2021) is used in full for Task 2, and the daily weather data (which extends to 2025) supplements the county-level wildfire data for trend analysis.

### 3.2 Task 1A: Wildfire Classification

**Primary model: Q-MoE Fire - Quantum Mixture-of-Experts (`04_proposed_algorithm/01_QMoE_Fire`)**

California spans 16 climate zones with fundamentally different fire dynamics. A single quantum model forced to learn all regimes simultaneously must average across them. Q-MoE instead routes each county-month to specialized quantum experts matched to its climate regime.

**Architecture**:

1. **Regime discovery**: KMeans clustering on county-level geography and weather (lat, long, avg temperature, humidity, wind, precipitation, fire rate) produces 4 natural climate regimes:
   - Regime A (24 counties): Moderate coastal/central California, 4.9% fire rate
   - Regime B (13 counties): Hot, fire-prone inland (Fresno, Kern, Los Angeles), 39% fire rate
   - Regime C (17 counties): Cool, wet northern (Humboldt, Shasta, Butte), 20% fire rate
   - Regime D (4 counties): Extreme desert (Imperial, Inyo, Riverside, San Bernardino), 24% fire rate

2. **Four quantum reservoir experts**: Each expert uses 4 qubits and 10 reservoir layers with fixed (non-trainable) random weights, but with genuinely different entanglement topologies:
   - Expert A: Ring topology (circular CNOT chain)
   - Expert B: Ladder topology (alternating even/odd CNOT pairs)
   - Expert C: Star topology (hub-and-spoke CNOTs)
   - Expert D: Full topology (all-to-all CNOTs)
   Each expert produces 11 observables (4 PauliZ + 4 PauliX + 3 nearest-neighbor ZZ correlations).

3. **Concat gating**: All 44 expert features (4 x 11) are concatenated and fed to a GradientBoosting head that learns which expert features matter for each sample. This outperformed explicit hard gating (regime-only) and soft gating (distance-weighted).

4. **Result: F1 = 0.782, AUC = 0.934**

**Multi-task improvement - RA-MQTF (`04_proposed_algorithm/04_RegimeAware_MQTF`)**: Adding a shared multi-task backbone on top of Q-MoE's expert features further improves fire classification to **F1 = 0.834** (+6.6%) and fire severity to **R2 = 0.639** (+19.4%). Joint training acts as regularization.

**Per-regime validation** - Q-MoE improves every climate regime (4/4):

| Regime | N_test | Fire Rate | MoE F1 | Baseline F1 | F1 Lift |
|--------|--------|-----------|--------|-------------|---------|
| A (coastal) | 570 | 7.4% | 0.693 | 0.582 | +0.111 |
| B (inland hot) | 308 | 41.9% | 0.833 | 0.736 | +0.098 |
| C (northern wet) | 402 | 22.9% | 0.739 | 0.680 | +0.059 |
| D (desert) | 94 | 24.5% | 0.818 | 0.708 | +0.110 |

**Concept and assumptions**: The core assumption is that different California climate zones exhibit different fire-weather relationships (validated by the per-regime ablation). The quantum reservoir acts as a non-linear feature extractor, and different entanglement topologies create genuinely different feature maps - analogous to using different convolutional filters in classical CNNs. The fixed (non-trainable) reservoir avoids barren plateaus entirely.

**Other quantum classifiers explored** (all on identical test splits):

| Model | F1 | Qubits | Notebook |
|-------|----|--------|----------|
| RA-MQTF (multi-task) | **0.834** | 4x4 | `04_proposed_algorithm/04_RegimeAware_MQTF` |
| Q-MoE Concat + GBM | 0.782 | 4x4 | `04_proposed_algorithm/01_QMoE_Fire` |
| QART (cross-attention) | 0.755 | 4 | `04_proposed_algorithm/02_QART` |
| QRC + LogReg | 0.724 | 8 | `02_quantum_baselines/03_Cloud_Quantum_Models` |
| MQTF (multi-task) | 0.693 | 4 | `03_novel_explorations/04_MultiTask_QTF` |
| QGR (graph) | 0.650 | 4 | `03_novel_explorations/02_Quantum_Graph_Reservoir` |
| Dressed Quantum Circuit | 0.637 | 4 | `02_quantum_baselines/04_Advanced_Architectures` |
| VQC (variational) | 0.542 | 6 | `01_eda_and_visualization/01_EDA_and_Baselines` |

**Forward-looking ZIP-code predictions**: County-level Q-MoE fire probabilities are mapped to ZIP codes using county-to-ZIP lookup, and combined with the domain-driven ensemble risk scoring to produce per-ZIP wildfire risk predictions for 2,174 California locations. This is an approximation; the proposed ZIP-STR-QMoE algorithm (Section 5) would perform this prediction directly at ZIP level.

### 3.3 Task 1B: Evaluation

**Advantages of our quantum approach**:
- **No barren plateaus**: Fixed quantum reservoirs (QRC, Q-MoE) have zero trainable quantum parameters, completely avoiding the barren plateau problem that undermines variational methods (VQC achieved only F1=0.542)
- **Regime specialization**: Different entanglement topologies create diverse feature maps that capture different fire dynamics - validated by per-regime improvements in all 4 climate zones
- **Scalable**: The 4-qubit reservoir runs efficiently on simulators and could be deployed on current NISQ hardware (IonQ Aria, Rigetti Ankaa)
- **Interpretable**: Classical heads (GBM, LogReg) provide feature importance; SHAP analysis identifies key wildfire drivers

**Disadvantages and limitations**:
- **Quantum overhead**: Encoding 8,584 samples x 4 experts takes ~7 minutes on a simulator - comparable to classical GBM training but with no parallelization advantage
- **Indirect quantum benefit**: The quantum reservoir is a non-linear feature extractor; the actual classification is performed by a classical head. The benefit comes from the feature space, not from quantum speedup
- **Circuit depth limitation**: At 10 layers x 4 qubits, the reservoir is shallow enough for NISQ devices, but deeper circuits might extract richer features (limited by decoherence)
- **County granularity**: Q-MoE and RA-MQTF operate at county level; ZIP-level predictions require a mapping step that loses spatial precision

**Classical comparison** (identical data, identical splits):

| Model | F1 | Type | Delta vs SVM |
|-------|-----|------|-----------|
| RA-MQTF | **0.834** | Quantum | +17.6% |
| Q-MoE Fire | 0.782 | Quantum | +10.3% |
| SVM (RBF) | 0.709 | Classical | - |
| XGBoost | 0.682 | Classical | - |
| Logistic Regression | 0.673 | Classical | - |
| Random Forest | 0.651 | Classical | - |

**Progression of quantum models**:
- Generation 1 (EDA notebook): VQC F1=0.542 - underperforms classical (barren plateaus)
- Generation 2 (Cloud QML notebook): QRC F1=0.724 - first improvement over classical (+2.1% over SVM)
- Generation 3 (Q-MoE notebook): Q-MoE F1=0.782 - regime-aware experts (+10.3% over SVM)
- Generation 4 (RA-MQTF notebook): RA-MQTF F1=0.834 - multi-task fusion (+17.6% over SVM)

### 3.4 Task 2: Insurance Premiums and Fire Severity

**ZIP-level premium prediction - QLSTM (Cloud Quantum Models notebook)**:
- 4-qubit variational circuits replace standard LSTM forget, input, cell, and output gates
- StronglyEntanglingLayers ansatz with 3 layers per gate (144 quantum parameters total)
- 2-step time sequences of ZIP-code insurance features predict next-year premium
- Input features include both the dataset's fire risk score and Q-MoE's Task 1 county-level fire probability (mapped to ZIP), directly connecting Task 1 output to Task 2
- **Result: R2 = 0.349 (log-scale temporal split), RMSE = 0.458**
- QLSTM substantially outperforms Classical LSTM (R2=-1.033) on the same temporal split, demonstrating that quantum gates capture temporal premium dynamics that standard recurrent cells miss
- Temporal XGBoost (R2=0.828) remains the strongest temporal baseline; Random Forest achieves R2=0.922 on a non-temporal split
- The premium prediction task is inherently difficult with only 4 years of annual ZIP data

**County-level fire severity - RA-MQTF (Regime-Aware MQTF notebook)**:
- Same Q-MoE expert features with multi-task backbone predicting log(burned acres)
- **Result: R2 = 0.639, RMSE = 1.563** - best severity prediction in the project
- Outperforms Q-MoE single-task (R2=0.535), MQTF (R2=0.475), and QART (R2=0.279)
- Multi-task regularization provides +19.4% improvement over single-task training

**Supporting explorations**:
- MQTF demonstrated that multi-task learning provides +69% F1 and +588% R2 over single-task baselines, motivating RA-MQTF
- Q-CAST simulated 6 climate scenarios through the quantum pipeline, showing that drought flips 35 county-months to fire-prone, with conformal 90% coverage intervals
- CQP provided conformal uncertainty quantification with ~90% empirical coverage

### 3.5 Quantum Resource Requirements

All models run on PennyLane `default.qubit` simulator. AWS Braket integration for IonQ Aria, Rigetti Ankaa-3, and IQM Garnet is implemented in the Cloud Quantum Models notebook.

| Model | Qubits | Circuit Depth | Trainable Params | Time (simulator) |
|-------|--------|---------------|-------------------|-----------------|
| Q-MoE (4 experts) | 4 x 4 | 40 each | 0 (fixed) + GBM | ~7 min |
| RA-MQTF (multi-task) | 4 x 4 | 40 each | 0 (fixed) + shared NN (6,595) | ~6.5 min |
| QLSTM (Task 2) | 4 | 12 | 144 | 15s |
| QRC (Task 1A baseline) | 8 | 60 | 0 (fixed) | 18s |

---

## Section 4: Results and Code Repository

### Results Summary

| Task | Model | Granularity | Score | Classical Best | Delta |
|------|-------|-------------|-------|----------------|-------|
| 1A Classification | RA-MQTF | County | **F1=0.834** | SVM F1=0.709 | +17.6% |
| 1A Classification | Q-MoE Fire | County | F1=0.782 | SVM F1=0.709 | +10.3% |
| 2 Fire Severity | RA-MQTF | County | **R2=0.639** | - | Best overall |
| 2 Premium (time-series) | QLSTM | ZIP | **R2=0.349** | LSTM R2=-1.033 | QLSTM wins |
| 2 Premium (time-series) | Temporal XGBoost | ZIP | R2=0.828 | - | Best temporal |
| 2 Premium (non-temporal) | Random Forest | ZIP | R2=0.922 | - | Non-temporal split |
| Forward Predictions | Q-MoE + ensemble | ZIP | 2,174 ZIPs | - | Per-ZIP risk map |

### Rubric-Aligned Summary

| Criterion | Evidence | Where |
|-----------|----------|-------|
| **Task 1A: Wildfire risk** | RA-MQTF F1=0.834 (county, mapped to ZIP) | `04_proposed_algorithm/04_RegimeAware_MQTF`, `05_predictions/01_ZipCode_2026_Predictions` |
| **Task 1B: Evaluation** | 15+ models benchmarked, ablations, per-regime validation, SHAP | Sections 3.2-3.3, all baseline notebooks |
| **Task 2: Premium/severity** | QLSTM R2=0.349 (premiums, ZIP, temporal log-scale; beats Classical LSTM R2=-1.033), RA-MQTF R2=0.639 (severity, county) | `02_quantum_baselines/03_Cloud_Quantum_Models`, `04_proposed_algorithm/04_RegimeAware_MQTF` |
| **Novel algorithm** | Q-MoE (regime experts) + RA-MQTF (multi-task fusion) | `04_proposed_algorithm/` |
| **Envisioned algorithm** | ZIP-STR-QMoE: validated ingredients moved to direct ZIP-level prediction | Section 5 |
| **Quantum hardware path** | 4-qubit fixed reservoirs; AWS Braket integration for IonQ/Rigetti/IQM | `02_quantum_baselines/03_Cloud_Quantum_Models` |
| **Known limitation** | Wildfire models are county-level, mapped to ZIPs. Proposed algorithm addresses this. | Sections 2, 5 |

### Notebooks (17 total, organized in 5 folders)

| Folder | Notebook | Purpose |
|--------|----------|---------|
| `01_eda_and_visualization/` | 01 EDA and Baselines | Classical + quantum baselines for Task 1A |
| | 02 Geospatial Visualization | Risk maps + quantum resource analysis |
| | 03 Clustering UMAP HDBSCAN | ZIP-code risk clustering into 5 tiers |
| | 04 FullHistory Wildfire Analysis | 40-year wildfire trend analysis |
| `02_quantum_baselines/` | 01 Task2 Insurance QML | Classical + quantum baselines for Task 2 |
| | 02 Improved Quantum Models | PCA, data re-uploading, hybrid ensembles |
| | 03 Cloud Quantum Models | QRC, QLSTM, Trainable Kernel, Transfer Learning |
| | 04 Advanced Quantum Architectures | QCNN, PQK, QGAN, Quanvolution, Dressed Circuit |
| `03_novel_explorations/` | 01 RAM-QRSRE | Regime-adaptive multi-scale QML |
| | 02 Quantum Graph Reservoir | Spatial graph + quantum reservoir |
| | 03 Conformal Quantum Prediction | Uncertainty quantification with coverage guarantees |
| | 04 Multi-Task Quantum Temporal Fusion | Joint fire + insurance prediction |
| `04_proposed_algorithm/` | **01 Q-MoE Fire** | Regime-diverse quantum experts (F1=0.782) |
| | **02 QART** | Quantum Adaptive Reservoir Transformer |
| | **03 Q-CAST** | Scenario analysis + decision support |
| | **04 RA-MQTF** | Multi-task fusion (F1=0.834, R2=0.639) |
| `05_predictions/` | 01 ZipCode Predictions | Final per-ZIP wildfire risk predictions using Q-MoE features |

### Code Repository

**https://github.com/Priti0427/Quantum-Sustainability-Challenge**


All notebooks, data, and results are reproducible. Executed notebooks with full cell outputs are included. See README.md for setup instructions and pipeline documentation.

---

## Section 5: Envisioned Algorithm

### ZIP-STR-QMoE: ZIP-level Spatio-Temporal Regime-aware Quantum Mixture-of-Experts

#### The gap this addresses

Our validated models (Q-MoE, RA-MQTF) produce strong wildfire predictions at county granularity but the challenge asks for ZIP-code-level wildfire predictions from historical data. Currently we bridge this gap by mapping county fire probabilities to ZIPs, which loses spatial precision. ZIP-STR-QMoE is designed to perform this prediction directly at ZIP level.

#### Architecture

ZIP-STR-QMoE extends our validated ingredients into a single ZIP-level system:

1. **ZIP-level regime routing** (validated by Q-MoE at county scale): Instead of clustering 58 counties into 4 regimes, cluster ~2,000 ZIP codes into 6-8 finer regimes using ZIP-level insurance features, CalFire risk scores, and local weather. Each ZIP-month is routed to the quantum expert matching its regime. Our Q-MoE results prove that regime routing improves all climate zones (4/4 regimes improved).

2. **Topology-diverse quantum reservoir experts** (validated by Q-MoE): The same 4-qubit fixed reservoirs with ring, ladder, star, and full entanglement topologies. These are the same circuits already validated on county data; no architectural change is needed, only retraining on ZIP-level features. Fixed reservoirs avoid barren plateaus entirely.

3. **Multi-task shared backbone** (validated by RA-MQTF): A shared neural network jointly predicts wildfire risk, fire severity, and insurance premiums. Our RA-MQTF results prove that multi-task training improves fire classification by +6.6% and severity by +19.4% over single-task baselines.

4. **Temporal QLSTM premium head** (validated by QLSTM): For the premium prediction task, replace the simple regression head with a QLSTM temporal module operating on 2-step ZIP-level insurance sequences. Our QLSTM results (R2=0.349, beating Classical LSTM R2=-1.033) prove quantum gates can learn temporal premium dynamics at ZIP granularity.

5. **Conformal uncertainty calibration** (validated by CQP): Wrap all predictions with adaptive conformal intervals providing 90% coverage guarantees. Our CQP results achieve ~90% empirical coverage.

#### What changes vs. current system

| Component | Current (county) | ZIP-STR-QMoE (ZIP) |
|-----------|------------------|---------------------|
| Granularity | 58 counties | ~2,000 ZIP codes |
| Regime clustering | 4 county regimes | 6-8 ZIP regimes |
| Input features | Weather + fire history | Weather + fire history + insurance + CalFire risk |
| Premium prediction | Mapped from county | Direct ZIP-level QLSTM |
| Quantum circuits | Same 4-qubit reservoirs | Same 4-qubit reservoirs |

#### Why we believe this will work

Every individual component is already validated with reproducible metrics:
- Regime routing improves all climate zones (Q-MoE, 4/4 regimes)
- Multi-task training improves fire classification +6.6% and severity +19.4% (RA-MQTF)
- QLSTM outperforms Classical LSTM on temporal premium prediction (R2=0.349 vs -1.033, Cloud Quantum Models notebook)
- Conformal calibration achieves ~90% coverage (CQP)

The main engineering challenge is the data pipeline: joining wildfire county data to ZIP-level insurance records requires a county-to-ZIP mapping layer during training. This is an implementation task, not an algorithmic uncertainty.

#### Quantum hardware requirements

- **Qubits**: 4 per expert x 4 experts = 16 logical qubits (feasible on IonQ Aria 25-qubit, IBM Eagle 127-qubit)
- **Circuit depth**: 40 layers per reservoir (shallow enough for NISQ with error mitigation)
- **Trainable quantum params**: 0 (fixed reservoirs) + 144 (QLSTM premium head)
- **Estimated time**: ~15 minutes on simulator for full ~2,000 ZIP encoding + training
- **Near-term path**: Fixed reservoirs are naturally noise-resilient; the QLSTM head is the only component requiring parameter optimization on quantum hardware

#### Why this matters

ZIP-STR-QMoE demonstrates that practical quantum benefit for sustainability applications comes from matching quantum architecture to problem structure: regime-diverse experts for climate heterogeneity, fixed reservoirs for noise resilience, and multi-task learning for the fire-insurance causal link. This pattern is transferable to other geospatial prediction problems (flood risk, crop yield, infrastructure resilience) wherever spatial heterogeneity, uncertainty, and multi-task structure coexist.
