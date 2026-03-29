# Quantum Sustainability Challenge

Quantum Machine Learning solution for the Deloitte Quantum Sustainability Challenge — predicting California wildfire risk and insurance premiums using hybrid quantum-classical models.

## Project Structure

```
├── data/                              # Datasets (wildfire + insurance)
├── notebooks/
│   ├── 01_EDA_and_Baselines.ipynb           # EDA + classical & quantum baselines (Task 1A)
│   ├── 02_Task2_Insurance_QML.ipynb         # Insurance premium prediction (Task 2)
│   ├── 03_Geospatial_Visualization.ipynb    # Maps, scalability analysis
│   ├── 04_Clustering_UMAP_HDBSCAN.ipynb     # ZIP-code risk clustering (5 tiers)
│   ├── 04_Improved_Quantum_Models.ipynb     # PCA, data re-uploading, hybrid ensemble
│   ├── 05_Cloud_Quantum_Models.ipynb        # QRC, QLSTM, Trainable Kernel, Transfer Learning
│   ├── 06_ZipCode_2026_Predictions.ipynb    # Final 2026 per-ZIP risk predictions
│   ├── 07_FullHistory_Wildfire_Analysis.ipynb # Extended 40-year wildfire analysis
│   ├── 08_Advanced_Quantum_Architectures.ipynb # QCNN, PQK, QGAN, Quanvolution, Dressed Circuit
│   ├── 09_Regime_Adaptive_Multiscale_QML.ipynb # RAM-QRSRE novel algorithm
│   ├── 10_Quantum_Adaptive_Reservoir_Transformer.ipynb # QART — core novel algorithm
│   ├── 11_Quantum_Graph_Reservoir.ipynb          # QGR — spatial graph + quantum reservoir
│   ├── 12_Conformal_Quantum_Prediction.ipynb     # CQP — conformal uncertainty quantification
│   ├── 13_MultiTask_Quantum_Temporal_Fusion.ipynb # MQTF — joint fire + insurance prediction
│   ├── 14_QMoE_Fire.ipynb                       # Q-MoE Fire — regime-aware quantum mixture-of-experts
│   └── 15_QCAST_Scenario_Transformer.ipynb      # Q-CAST — forward-looking "what if" scenario analysis
├── results/                           # Plots, CSVs, JSON outputs, interactive HTML maps
├── requirements.txt                   # Python dependencies
└── README.md
```

## Tasks and Best Results

| Task | Description | Best Model | Score | Type |
|------|-------------|-----------|-------|------|
| 1A | Wildfire classification (county) | Q-MoE Fire (4×4 qubits) | **F1 = 0.782** | Quantum |
| 1A | Wildfire classification (county) | QART (4 qubits) | F1 = 0.755 | Quantum |
| 1A | Wildfire classification (daily) | QRC + LogReg (8 qubits) | F1 = 0.724 | Quantum |
| 1A | Wildfire classification (daily) | SVM (RBF) | F1 = 0.709 | Classical |
| 2 | Fire severity regression (county) | Q-MoE Fire (4×4 qubits) | **R² = 0.535** | Quantum |
| 2 | Fire severity regression (county) | MQTF Multi-Task (4 qubits) | R² = 0.475 | Quantum |
| 2 | Insurance premium time series | QLSTM (4 qubits) | **R² = 0.922** | Quantum |
| 2 | Insurance premium regression | Random Forest | R² = 0.922 | Classical |
| — | ZIP-code risk clustering | UMAP + HDBSCAN | 1,829 ZIPs in 5 tiers | Unsupervised |
| — | 2026 predictions | Ensemble + domain features | 2,174 ZIPs scored | Forecast |
| — | Scenario analysis | Q-CAST (6 scenarios) | Decision-support | Quantum |

**Fair time-series comparison (identical data splits):** QLSTM R²=0.922 vs Classical LSTM R²=0.765 vs Temporal XGBoost R²=0.860.

**Note on Task 2 metrics:** Earlier versions used premium-derived features (Earned Premium, Premium Per Policy, Loss Ratio) to predict next-year premium, which constituted data leakage (R² was artificially 0.99+). These features were removed — the scores above reflect realistic predictive power using only risk, exposure, and loss features.

## Quantum Techniques Used

**Primary (validated with reproducible metrics):**
- **Quantum Reservoir Computing (QRC)** — 8 qubits, 15 fixed random layers, avoids barren plateaus. Best standalone quantum classifier (F1=0.724).
- **Quantum LSTM (QLSTM)** — 4 qubits, PyTorch + PennyLane. True time-series QML. Outperforms classical LSTM and XGBoost on identical data.
- **Quantum Kernel SVM** — 6 qubits, fidelity kernel. F1=0.683.

**Supporting:**
- Variational Quantum Classifier (VQC) with data re-uploading
- Variational Quantum Regressor (VQR) with log-transformed targets
- PCA preprocessing for decorrelated quantum encoding
- Hybrid quantum-classical stacking ensembles
- Quantum Convolutional Neural Network (QCNN) — 8 qubits, conv+pool layers
- Projected Quantum Kernel (PQK) — local Pauli shadows + RBF kernel
- Quantum GAN for data augmentation — generates synthetic fire-day samples
- Quanvolutional Neural Network — quantum feature extraction + classical MLP
- Dressed Quantum Circuit — alternating classical-quantum sandwich architecture

**Novel Algorithms (core competition entry):**
- **QART (Quantum Adaptive Reservoir Transformer)** — multi-scale quantum reservoirs + quantum cross-attention fusion + data re-uploading heads. Unified architecture for both Task 1A and Task 2.
- RAM-QRSRE (Regime-Adaptive Multi-Scale Quantum Reservoir with Re-uploading Experts) — multi-scale reservoirs + regime-routed residual quantum heads
- **QGR (Quantum Graph Reservoir)** — county adjacency graph + quantum reservoir node encoding + graph message passing for spatially-aware wildfire prediction
- **CQP (Conformal Quantum Prediction)** — adaptive conformal prediction on QRC outputs for calibrated uncertainty intervals with formal coverage guarantees
- **MQTF (Multi-Task Quantum Temporal Fusion)** — shared quantum reservoir backbone with three prediction heads (fire classification, fire severity, insurance premium) and conformal uncertainty
- **Q-MoE Fire (Quantum Mixture-of-Experts for Climate Regimes)** — 4 quantum reservoir experts with different entanglement topologies routed by KMeans climate regime clustering, soft/hard gating, per-regime specialization
- **Q-CAST (Quantum Causal Scenario Transformer)** — forward-looking "what if" risk simulation: 6 climate/mitigation scenarios through quantum reservoir + conformal intervals, per-county impact analysis, decision-support dashboard

**Exploratory (code written, results in Notebook 05):**
- Trainable Quantum Kernel (kernel-target alignment optimization)
- Quantum Transfer Learning (classical MLP + quantum head)
- AWS Braket real hardware integration (IonQ Aria, Rigetti Ankaa, IQM Garnet)

## Setup

```bash
pip install -r requirements.txt
```

## Tech Stack

- **PennyLane** + PyTorch (QRC, QLSTM, Transfer Learning, Trainable Kernels, QCNN, PQK, QGAN, Quanvolution, Dressed Circuit, QGR, CQP, MQTF, Q-MoE, Q-CAST)
- **Qiskit** (VQC, VQR, quantum kernels)
- **scikit-learn, XGBoost** (classical baselines, ensembles)
- **SHAP** (model explainability — beeswarm, waterfall, and dependence plots)
- **Optuna** (hyperparameter tuning with cross-validation)
- **UMAP, HDBSCAN** (unsupervised clustering)
- **pandas, matplotlib, seaborn, Plotly** (data analysis, visualization)

## Quantum Resource Requirements

| Model | Qubits | Circuit Depth | Trainable Params | Training Time |
|-------|--------|--------------|-------------------|---------------|
| QRC (Task 1A) | 8 | 60 | 0 (fixed) | 18s |
| QRC (Task 2) | 8 | 60 | 0 (fixed) | 13s |
| QLSTM | 4 | 12 | 144 | 15s |
| Trainable Kernel | 6 | 12 | 12 | ~39min |
| Transfer Learning | 4 | 8 | 24 + MLP | <1s |
| QCNN | 8 | ~20 | 63 | ~varies |
| Projected Quantum Kernel | 6 | ~8 | 0 | ~varies |
| Quantum GAN (Generator) | 6 | ~12 | 54 + MLP | ~varies |
| Quanvolutional NN | 4 | ~8 | quantum + MLP | ~varies |
| Dressed Quantum Circuit | 4 | ~12 x2 | quantum + MLP | ~varies |
| QART Reservoir Bank (x3) | 4 | 4/6/8 | 0 (fixed) | ~varies |
| QART Attention + Head | 4 | ~6 | quantum + MLP | ~varies |
| QGR Reservoir (graph) | 4 | 40 | 0 (fixed) + graph | ~varies |
| CQP (conformal wrapper) | 4 | 40 | 0 (fixed) + quantile | ~varies |
| MQTF (multi-task heads) | 4 | 40 | 0 (fixed) + 3 heads | ~varies |
| Q-MoE (4 experts) | 4 x 4 | 40 each | 0 (fixed) + gating | ~varies |
| Q-CAST (scenario engine) | 4 | 40 | 0 (fixed) + heads | ~varies (x6 scenarios) |

## End-to-End Pipeline

Run the notebooks in this order. Each notebook reads from `data/` and writes results to `results/`.

```
01_EDA_and_Baselines          → Classical & quantum baselines for Task 1A
    ↓                            Outputs: classical_baselines.json, SHAP plots
02_Task2_Insurance_QML        → Classical & quantum baselines for Task 2
    ↓                            Outputs: task2_results.csv, SHAP plots
03_Geospatial_Visualization   → Risk maps + quantum resource analysis
    ↓                            Outputs: california_fire_risk_map.png, scalability plots
04_Clustering_UMAP_HDBSCAN    → ZIP-code risk clustering into 5 tiers
    ↓                            Outputs: zip_cluster_labels.csv, zip_umap_embedding.csv
04_Improved_Quantum_Models    → PCA, re-uploading, hybrid ensembles
    ↓                            Outputs: improved_results.csv
05_Cloud_Quantum_Models       → QRC, QLSTM, Trainable Kernel, Transfer Learning
    ↓                            Outputs: advanced_quantum_results.json, resource table
06_ZipCode_2026_Predictions   → Final per-ZIP risk predictions for 2026
    ↓                            Outputs: zip_risk_predictions_2026.csv, executive_dashboard.png
07_FullHistory_Wildfire_Analysis → 40-year wildfire trend analysis, concept drift detection
                                 Outputs: climate_trends_40yr.png, seasonal analysis
08_Advanced_Quantum_Architectures → QCNN, PQK, QGAN, Quanvolution, Dressed Circuit
                                 Outputs: nb08_model_comparison.png, nb08_advanced_architectures_results.json
09_Regime_Adaptive_Multiscale_QML → RAM-QRSRE novel algorithm
                                 Outputs: ram_qrsre_results.json, ram_qrsre_ablation.csv
10_Quantum_Adaptive_Reservoir_Transformer → QART — core novel algorithm (both tasks)
                                 Outputs: qart_results.json, qart_ablation.csv, qart_comparison.png
11_Quantum_Graph_Reservoir       → QGR — spatial graph + quantum reservoir (county-level)
                                 Outputs: qgr_results.json, qgr_classification_overview.png
12_Conformal_Quantum_Prediction  → CQP — conformal uncertainty on QRC (coverage guarantees)
                                 Outputs: cqp_results.json, cqp_calibration.png, cqp_intervals.png
13_MultiTask_Quantum_Temporal_Fusion → MQTF — joint fire + insurance with shared backbone
                                 Outputs: mqtf_results.json, mqtf_comparison.png, mqtf_conformal.png
14_QMoE_Fire                        → Q-MoE Fire — climate-regime-aware quantum mixture-of-experts
                                 Outputs: qmoe_results.json, qmoe_overview.png
15_QCAST_Scenario_Transformer       → Q-CAST — forward-looking "what if" scenario simulation
                                 Outputs: qcast_results.json, qcast_dashboard.png, qcast_heatmap.png
```

All notebooks are committed with full cell outputs.

### Data Assumptions for 2026 Predictions

**Known features** (publicly available):
- ZIP-code fire risk scores from CalFire risk maps
- Historical premiums, exposures, loss ratios from CDI regulatory filings (2018–2021)

**Extrapolated features** (estimated):
- 2022–2025 premium trends (linear extrapolation from 2018–2021)
- Future fire loss amounts (from risk scores + historical distributions)

**Validation**: Temporal train/test split — train on 2018–2020, test on 2021. The 2026 predictions are extrapolations with confidence intervals, not validated forecasts.

### Model Explainability

SHAP (SHapley Additive exPlanations) is used across notebooks to provide transparent feature attribution:
- **NB01**: Beeswarm summary + waterfall plots for wildfire day predictions
- **NB02**: Beeswarm summary + dependence plot for premium drivers
- **NB06**: Beeswarm summary + top-5 riskiest ZIP waterfall breakdowns

All SHAP visualizations are saved as PNGs in `results/` and embedded in notebook outputs.
