# Quantum Sustainability Challenge

Quantum Machine Learning solution for the Deloitte Quantum Sustainability Challenge — predicting California wildfire risk and insurance premiums using hybrid quantum-classical models.

## Project Structure

```
├── data/                              # Datasets (wildfire + insurance)
├── notebooks/
│   ├── 01_EDA_and_Baselines.ipynb           # EDA + classical & quantum baselines (Task 1A)
│   ├── 02_Task2_Insurance_QML.ipynb         # Insurance premium prediction (Task 2)
│   ├── 03_Geospatial_Visualization.ipynb    # Maps, scalability analysis
│   ├── 04_Improved_Quantum_Models.ipynb     # PCA, data re-uploading, hybrid ensemble
│   ├── 04_Clustering_UMAP_HDBSCAN.ipynb     # ZIP-code risk clustering (5 tiers)
│   ├── 05_Cloud_Quantum_Models.ipynb        # QRC, QLSTM, Trainable Kernel, Transfer Learning
│   └── 06_ZipCode_2026_Predictions.ipynb    # Final 2026 per-ZIP risk predictions
├── results/                           # Plots, CSVs, and JSON outputs
├── requirements.txt                   # Python dependencies
└── README.md
```

## Tasks and Best Results

| Task | Description | Best Model | Score | Type |
|------|-------------|-----------|-------|------|
| 1A | Wildfire day classification | QRC + LogReg (8 qubits) | **F1 = 0.724** | Quantum |
| 1A | Wildfire day classification | Hybrid Ensemble (XGB+RF+LR+VQC) | F1 = 0.705 | Hybrid |
| 2 | Insurance premium time series | QLSTM (4 qubits) | **R² = 0.922** | Quantum |
| 2 | Insurance premium regression | Hybrid Ensemble (XGB+RF+LinReg+VQR) | R² = 0.998 | Hybrid |
| — | ZIP-code risk clustering | UMAP + HDBSCAN | 1,829 ZIPs in 5 tiers | Unsupervised |
| — | 2026 predictions | Ensemble + domain features | 2,174 ZIPs scored | Forecast |

**Fair time-series comparison (identical data splits):** QLSTM R²=0.922 vs Classical LSTM R²=0.765 vs XGBoost R²=0.860.

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

**Exploratory (code written, results in Notebook 05):**
- Trainable Quantum Kernel (kernel-target alignment optimization)
- Quantum Transfer Learning (classical MLP + quantum head)
- AWS Braket real hardware integration (IonQ Aria, Rigetti Ankaa, IQM Garnet)

## Setup

```bash
pip install -r requirements.txt
```

## Tech Stack

- **PennyLane** + PyTorch (QRC, QLSTM, Transfer Learning, Trainable Kernels)
- **Qiskit** (VQC, VQR, quantum kernels)
- **scikit-learn, XGBoost** (classical baselines, ensembles)
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

## End-to-End Pipeline

Run the notebooks in this order. Each notebook reads from `data/` and writes results to `results/`.

```
01_EDA_and_Baselines          → Classical & quantum baselines for Task 1A
    ↓                            Outputs: classical_baselines.json, plots
02_Task2_Insurance_QML        → Classical & quantum baselines for Task 2
    ↓                            Outputs: task2_results.csv, plots
03_Geospatial_Visualization   → Risk maps + quantum resource analysis
    ↓                            Outputs: california_fire_risk_map.png, scalability plots
04_Improved_Quantum_Models    → PCA, re-uploading, hybrid ensembles
    ↓                            Outputs: improved_results.csv, model_comparison.csv
04_Clustering_UMAP_HDBSCAN    → ZIP-code risk clustering into 5 tiers
    ↓                            Outputs: zip_cluster_labels.csv, zip_umap_embedding.csv
05_Cloud_Quantum_Models       → QRC, QLSTM, Trainable Kernel, Transfer Learning
    ↓                            Outputs: advanced_quantum_results.json, resource table
06_ZipCode_2026_Predictions   → Final per-ZIP risk predictions for 2026
                                 Outputs: zip_risk_predictions_2026.csv, executive_dashboard.png
```

Pre-executed notebooks (`*_executed.ipynb`) are included with full cell outputs.

### Data Assumptions for 2026 Predictions

**Known features** (publicly available):
- ZIP-code fire risk scores from CalFire risk maps
- Historical premiums, exposures, loss ratios from CDI regulatory filings (2018-2021)

**Extrapolated features** (estimated):
- 2022-2025 premium trends (linear extrapolation from 2018-2021)
- Future fire loss amounts (from risk scores + historical distributions)

**Validation**: Temporal train/test split — train on 2018-2020, test on 2021. The 2026 predictions are extrapolations with confidence intervals, not validated forecasts.
