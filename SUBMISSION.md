# Deloitte Quantum Sustainability Challenge — Submission

---

## Section 1: Team Overview

**Team Name**: Quantum Wildfire Analytics

| Member | University | Email |
|--------|-----------|-------|
| Priti Sagar | Drexel University | ps3648@drexel.edu |
| Gautam Mahajan | University of Connecticut — Storrs | gautam.mahajan@uconn.edu |

---

## Section 2: Abstract (ONE page, ≤400 words)

We present a hybrid quantum-classical machine learning system for predicting California wildfire risk and mapping it to insurance premiums, addressing Tasks 1A, 1B, and 2.

Our work systematically evaluates 15+ quantum architectures across 15 notebooks, progressing from standard models (VQC, Quantum Kernels) through advanced architectures (QCNN, Quantum GANs) to seven novel algorithms designed for this challenge. The final system combines climate-regime-aware quantum mixture-of-experts with forward-looking scenario analysis, uncertainty quantification, and multi-task learning.

**Task 1A — Wildfire Classification**: Our best model, Q-MoE Fire (Quantum Mixture-of-Experts for Climate Regimes), achieves F1=0.782 and AUC=0.934. Q-MoE recognizes that California is not one homogeneous climate system. It clusters 58 counties into 4 climate regimes, then routes each sample to 4 specialized quantum reservoir experts with different entanglement topologies (ring, ladder, star, full). Q-MoE outperforms all classical baselines (SVM F1=0.709) and all single-model quantum approaches (QART F1=0.755, QRC F1=0.724). Per-regime ablation confirms that every climate zone benefits from specialization.

**Task 1B — Evaluation**: Quantum advantage is architecture-dependent. Variational methods (VQC F1=0.542) suffer barren plateaus; fixed reservoirs with classical readouts (QRC F1=0.724) avoid them; regime-aware expert routing (Q-MoE F1=0.782) outperforms both. The key insight: quantum advantage comes from matching architecture to problem structure, not from deeper circuits.

**Task 2 — Insurance Premiums**: QLSTM achieves R²=0.922 for ZIP-level premium time-series, outperforming Classical LSTM (R²=0.765) and XGBoost (R²=0.860). For county-level fire severity, Q-MoE achieves R²=0.535.

**Forward-Looking Decision Support**: Q-CAST (Quantum Causal Scenario Transformer) converts our model into a "what-if" engine — simulating 6 climate scenarios (drought, extreme wind, compound crisis, controlled burns, wet year) through the quantum pipeline. Under drought (-30% precipitation, +2°F), 35 county-months flip from safe to fire-prone. Conformal prediction provides 90% coverage intervals on all scenario outputs.

**2026 Predictions**: Per-ZIP wildfire risk scores are produced for 2,174 California locations using quantum-derived features and domain-driven indicators (NB06).

**Envisioned next-generation algorithm**: C-STQGR (Conformal Spatio-Temporal Quantum Graph Reservoir) would unify graph structure, conformal calibration, multi-task learning, and regime-aware expert routing into a single ZIP-code-scale system, targeting real-time quantum hardware deployment.

We propose novel quantum architectures specifically designed for wildfire-insurance modeling. While our work builds on established quantum computing primitives (quantum reservoir computing, mixture-of-experts, conformal prediction), the specific architectural designs — topology-diversified quantum expert routing (Q-MoE), multi-scale reservoir cross-attention fusion (QART), and quantum scenario simulation for insurance decision support (Q-CAST) — are original contributions with no direct precedent in the literature.

---

## Section 3: Detailed Algorithm Description

### 3.1 Data Pipeline

**Challenge-provided datasets**:
- **Wildfire data**: California wildfire records including fire name, location, county, burned acreage (GIS_ACRES), organized at monthly county-level granularity (10,989 rows, 58 counties, 2008–2020)
- **Insurance data**: California Department of Insurance homeowners insurance filings — `insurance_2018_2019.XLS` and `insurance_2020_2021.XLS` — containing ZIP-level premiums, exposures, losses, and risk scores (~2,000 ZIPs per year)

**Additional data used**:
- **Daily weather data**: Temperature (min/max/avg), humidity, wind speed, precipitation, snow, sun hours — merged with fire occurrence at daily and monthly granularity. Source: aggregated weather station data included in `wildfire_county_monthly.csv`
- **Historical wildfire data (1984–2025)**: Extended CalFire incident archive for long-term trend analysis (NB07)
- **CalFire risk maps**: ZIP-code-level fire risk scores used for 2026 predictions (NB06)

No external APIs or real-time data feeds were used. All data is included in the repository.

**Feature engineering**: Temperature range, wind-temperature ratio, lagged precipitation and wind speed, PCA-reduced quantum encodings (4 components), county adjacency graph features, temporal lag features (lag_acres, lag_temp), and domain-driven indicators (Fire Weather Index proxy, drought severity, hydroclimate whiplash signals).

**Temporal split**: Train on 2008–2018, test on 2019–2020 (county-level); train on 2018–2020, test on 2021 (ZIP-level insurance).

### 3.2 Task 1A: Wildfire Classification

**Primary model: Q-MoE Fire — Quantum Mixture-of-Experts (NB14)**

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

3. **Concat gating**: All 44 expert features (4 × 11) are concatenated and fed to a GradientBoosting head that learns which expert features matter for each sample. This outperformed explicit hard gating (regime-only) and soft gating (distance-weighted).

4. **Result: F1 = 0.782, AUC = 0.934**

**Per-regime validation** — Q-MoE improves every climate regime (4/4):

| Regime | N_test | Fire Rate | MoE F1 | Baseline F1 | F1 Lift |
|--------|--------|-----------|--------|-------------|---------|
| A (coastal) | 570 | 7.4% | 0.693 | 0.582 | +0.111 |
| B (inland hot) | 308 | 41.9% | 0.833 | 0.736 | +0.098 |
| C (northern wet) | 402 | 22.9% | 0.739 | 0.680 | +0.059 |
| D (desert) | 94 | 24.5% | 0.818 | 0.708 | +0.110 |

**Concept and assumptions**: The core assumption is that different California climate zones exhibit different fire-weather relationships (validated by the per-regime ablation). The quantum reservoir acts as a non-linear feature extractor, and different entanglement topologies create genuinely different feature maps — analogous to using different convolutional filters in classical CNNs. The fixed (non-trainable) reservoir avoids barren plateaus entirely.

**Other quantum classifiers explored** (all on identical test splits):

| Model | F1 | Qubits | Notebook |
|-------|----|--------|----------|
| Q-MoE Concat + GBM | **0.782** | 4×4 | NB14 |
| QART (cross-attention) | 0.755 | 4 | NB10 |
| QRC + LogReg | 0.724 | 8 | NB05 |
| MQTF (multi-task) | 0.693 | 4 | NB13 |
| QGR (graph) | 0.650 | 4 | NB11 |
| Dressed Quantum Circuit | 0.637 | 4 | NB08 |
| Quanvolutional NN | 0.522 | 4 | NB08 |
| VQC (variational) | 0.542 | 6 | NB01 |

**2026 ZIP-code predictions**: County-level Q-MoE predictions are mapped to ZIP codes using county-to-ZIP lookup, and combined with the domain-driven ensemble risk scoring from NB06 to produce per-ZIP wildfire risk predictions for 2,174 California locations. Top-risk areas: Shasta County (0.96), Lassen County (0.88), Humboldt County (0.86), Sonoma County (0.82).

### 3.3 Task 1B: Evaluation

**Advantages of our quantum approach**:
- **No barren plateaus**: Fixed quantum reservoirs (QRC, Q-MoE) have zero trainable quantum parameters, completely avoiding the barren plateau problem that undermines variational methods (VQC achieved only F1=0.542)
- **Regime specialization**: Different entanglement topologies create diverse feature maps that capture different fire dynamics — validated by per-regime improvements in all 4 climate zones
- **Scalable**: The 4-qubit reservoir runs efficiently on simulators and could be deployed on current NISQ hardware (IonQ Aria, Rigetti Ankaa)
- **Interpretable**: Classical heads (GBM, LogReg) provide feature importance; SHAP analysis identifies key wildfire drivers

**Disadvantages and limitations**:
- **Quantum overhead**: Encoding 8,584 samples × 4 experts takes ~7 minutes on a simulator — comparable to classical GBM training but with no parallelization advantage
- **Indirect quantum benefit**: The quantum reservoir is a non-linear feature extractor; the actual classification is performed by a classical head. The quantum advantage comes from the feature space, not from quantum speedup
- **Circuit depth limitation**: At 10 layers × 4 qubits, the reservoir is shallow enough for NISQ devices, but deeper circuits might extract richer features (limited by decoherence)
- **County granularity**: Q-MoE operates at county level; ZIP-level predictions require a mapping step that loses spatial precision

**Classical comparison** (identical data, identical splits):

| Model | F1 | Type | Advantage |
|-------|-----|------|-----------|
| Q-MoE Fire | **0.782** | Quantum | +10.3% over best classical |
| SVM (RBF) | 0.709 | Classical | Best classical baseline |
| XGBoost | 0.682 | Classical | — |
| Logistic Regression | 0.673 | Classical | — |
| Random Forest | 0.651 | Classical | — |

**Progression of quantum advantage**:
- Generation 1 (NB01): VQC F1=0.542 — worse than classical (barren plateaus)
- Generation 2 (NB05): QRC F1=0.724 — first quantum advantage (+2.1% over SVM)
- Generation 3 (NB10): QART F1=0.755 — quantum attention fusion
- Generation 4 (NB14): Q-MoE F1=0.782 — regime-aware experts (+10.3% over SVM)

### 3.4 Task 2: Insurance Premiums and Fire Severity

**ZIP-level premium prediction — QLSTM (NB05)**:
- 4-qubit variational circuits replace standard LSTM forget, input, cell, and output gates
- StronglyEntanglingLayers ansatz with 3 layers per gate (144 quantum parameters total)
- 2-step time sequences of ZIP-code insurance features predict next-year premium
- **Result: R² = 0.922, RMSE = 0.158**
- Outperforms Classical LSTM (R²=0.765) and Temporal XGBoost (R²=0.860) on identical data

**County-level fire severity — Q-MoE Fire (NB14)**:
- Same 4-expert architecture predicting log(burned acres) instead of fire occurrence
- **Result: R² = 0.535, RMSE = 1.774** — best quantum regression on county data
- Beats MQTF (R²=0.475), single-expert QRC (R²=0.443), and QART (R²=0.279)

**Multi-task joint prediction — MQTF (NB13)**:
- Shared quantum reservoir backbone with three simultaneous prediction heads: fire classification, fire severity, insurance premium
- Joint learning provides massive improvements for fire tasks: classification F1 +69%, severity R² +588% compared to single-task baselines
- Fire severity R²=0.475 — second-best quantum regression
- Insurance premium prediction is limited by data granularity (only 4 years of annual ZIP data)

**Scenario-based premium sensitivity — Q-CAST (NB15)**:
- 6 climate scenarios simulated through the full quantum pipeline
- Drought scenario: +0.88% average probability increase, 35 county-months flip to fire-prone
- Per-county impact analysis identifies which counties need premium adjustments under each scenario
- Conformal prediction provides 90% coverage intervals on all scenario-conditioned predictions

### 3.5 Quantum Resource Requirements

All models run on PennyLane `default.qubit` simulator. AWS Braket integration for IonQ Aria, Rigetti Ankaa-3, and IQM Garnet is implemented in NB05.

| Model | Qubits | Circuit Depth | Trainable Params | Time (simulator) |
|-------|--------|---------------|-------------------|-----------------|
| Q-MoE (4 experts) | 4 × 4 | 40 each | 0 (fixed) + GBM | ~7 min |
| QART (3 reservoirs) | 4 | 4/6/8 + attn | quantum + MLP | ~4.5 min |
| QRC (Task 1A baseline) | 8 | 60 | 0 (fixed) | 18s |
| QLSTM (Task 2) | 4 | 12 | 144 | 15s |
| Q-CAST (6 scenarios) | 4 | 40 | 0 (fixed) + heads | ~8 min |
| QGR (graph reservoir) | 4 | 40 | 0 (fixed) + graph | ~2 min |
| CQP (conformal) | 4 | 40 | 0 (fixed) + quantile | ~2 min |
| MQTF (multi-task) | 4 | 40 | 0 (fixed) + 3 heads | ~2 min |

---

## Section 4: Results and Code Repository

### Results Summary

| Task | Best Quantum Model | Score | Classical Best | Quantum Advantage |
|------|-------------------|-------|----------------|-------------------|
| 1A Classification | Q-MoE Fire (4×4 qubits) | **F1=0.782** | SVM F1=0.709 | +10.3% |
| 1A Classification | QART (4 qubits) | F1=0.755 | SVM F1=0.709 | +6.5% |
| 2 Fire Severity | Q-MoE Fire (4×4 qubits) | **R²=0.535** | — | Best quantum |
| 2 Premium Time Series | QLSTM (4 qubits) | **R²=0.922** | XGBoost R²=0.860 | +7.2% |
| Uncertainty | CQP (conformal) | 90% coverage | — | Calibrated intervals |
| Scenario Analysis | Q-CAST (6 scenarios) | Decision-support | — | Forward-looking |
| 2026 Predictions | Ensemble + quantum features | 2,174 ZIPs scored | — | Per-ZIP risk map |

### Notebooks (15 total)

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | EDA and Baselines | Classical + quantum baselines for Task 1A |
| 02 | Task 2 Insurance QML | Classical + quantum baselines for Task 2 |
| 03 | Geospatial Visualization | Risk maps + quantum resource analysis |
| 04 | Clustering UMAP HDBSCAN | ZIP-code risk clustering into 5 tiers |
| 04 | Improved Quantum Models | PCA, data re-uploading, hybrid ensembles |
| 05 | Cloud Quantum Models | QRC, QLSTM, Trainable Kernel, Transfer Learning |
| 06 | ZipCode 2026 Predictions | Final per-ZIP risk predictions for 2026 |
| 07 | FullHistory Wildfire Analysis | 40-year wildfire trend analysis |
| 08 | Advanced Quantum Architectures | QCNN, PQK, QGAN, Quanvolution, Dressed Circuit |
| 09 | Regime Adaptive Multiscale QML | RAM-QRSRE novel algorithm |
| 10 | **QART** | Quantum Adaptive Reservoir Transformer |
| 11 | **QGR** | Quantum Graph Reservoir (spatial graph) |
| 12 | **CQP** | Conformal Quantum Prediction (uncertainty) |
| 13 | **MQTF** | Multi-Task Quantum Temporal Fusion |
| 14 | **Q-MoE Fire** | Quantum Mixture-of-Experts (best model) |
| 15 | **Q-CAST** | Quantum Causal Scenario Transformer |

### Code Repository

**https://github.com/Priti0427/Quantum-Sustainability-Challenge**

All notebooks, data, and results are reproducible. Pre-executed notebooks with full cell outputs are included. See README.md for setup instructions and pipeline documentation.

---

## Section 5: Envisioned Algorithm

### C-STQGR: Conformal Spatio-Temporal Quantum Graph Reservoir

We envision a next-generation algorithm that unifies the five building blocks we have individually validated in this project into a single end-to-end system operating at ZIP-code scale.

**Architecture**:

The envisioned C-STQGR (Conformal Spatio-Temporal Quantum Graph Reservoir) integrates:

1. **Quantum Graph Reservoir Layer** (proven in NB11): Encode each ZIP code's features through a quantum reservoir, then propagate information along a spatial graph whose edges represent geographic proximity, shared fire corridors, wind patterns, and vegetation continuity. Our NB11 results show that graph structure improves F1 by +0.059 and R² by +0.073 over non-graph approaches.

2. **Regime-Aware Quantum Mixture-of-Experts** (proven in NB14): Route each ZIP-month through specialized quantum experts matched to its climate zone. Our NB14 results demonstrate that different entanglement topologies create genuinely different feature maps, and specialization improves every climate regime. At ZIP-code scale (~2,000 nodes), the gating network would learn finer-grained regimes than our 4-county clusters.

3. **Multi-Task Temporal Fusion** (proven in NB13): Jointly predict wildfire risk, fire severity, and insurance premiums using a shared quantum backbone with task-specific heads. Our NB13 results show that multi-task learning provides +69% F1 improvement and +588% R² improvement over single-task baselines, because fire risk and premium dynamics are causally linked.

4. **Conformal Uncertainty Quantification** (proven in NB12): Wrap all predictions with adaptive conformal prediction intervals that provide formal coverage guarantees. Our NB12 results achieve ~90% empirical coverage with intervals 26–31% narrower than standard fixed-width approaches. For insurers, this transforms point predictions into trustworthy risk bands.

5. **Scenario Simulation Engine** (proven in NB15): Apply forward-looking "what if" climate perturbations through the full quantum pipeline. Our NB15 results demonstrate that drought scenarios flip 35 county-months to fire-prone and that the quantum reservoir captures non-trivial climate interactions (e.g., wind sometimes reduces fire probability in certain regions by disrupting buildup patterns).

**Expected benefits**:

- **Unified prediction**: A single model simultaneously forecasts wildfire occurrence, fire severity, insurance premium changes, and scenario-conditioned risk shifts — currently requiring 4 separate models
- **Spatial awareness**: Graph propagation captures fire corridor effects (e.g., Santa Ana wind corridors linking inland and coastal ZIP codes) that independent-ZIP models miss entirely
- **Calibrated uncertainty**: Every prediction comes with a coverage-guaranteed interval, enabling insurers to price risk bands rather than point estimates
- **Forward-looking**: Scenario analysis transforms the model from retrospective prediction to proactive catastrophe modeling, which is what California's insurance regulator explicitly requires
- **Climate-regime specialization**: Different quantum experts for coastal fog, inland heat, mountain snow, and desert regions capture the true heterogeneity of California's 16 climate zones

**Quantum hardware requirements**:

- **Qubits**: 4–8 per expert × 4–6 experts = 16–48 logical qubits (feasible on current NISQ devices: IonQ Aria has 25 qubits, IBM Eagle has 127)
- **Circuit depth**: 40–80 layers per reservoir (challenging for current coherence times, but feasible with error mitigation)
- **Shots**: ~1,000 per sample for Pauli expectation values (standard for current hardware)
- **Estimated wall-clock time**: ~30 minutes on a 25-qubit trapped-ion device for full pipeline (encoding + graph propagation + prediction)
- **Near-term path**: The reservoir components (zero trainable quantum parameters) are naturally noise-resilient, making C-STQGR a strong candidate for early fault-tolerant quantum advantage

**Why this matters for the quantum community**:

C-STQGR demonstrates that quantum advantage in real-world applications comes not from raw computational speedup but from architectural innovation — combining quantum feature spaces with classical graph structure, conformal calibration, and domain-specific expert routing. This pattern is transferable to other sustainability problems (flood prediction, crop yield forecasting, supply chain resilience) where spatial structure, uncertainty, and regime heterogeneity all matter.

The individual components are validated in our 15 notebooks. The envisioned unified system represents the natural next step as quantum hardware scales from 25 to 100+ qubits with improved coherence times.
