# Quantum Sustainability Challenge 2026 

Hybrid quantum-classical solution for the **Deloitte Quantum Sustainability Challenge 2026**: predicting California wildfire risk and homeowners insurance premiums using regime-aware quantum mixture-of-experts, multi-task fusion, and quantum LSTM models.

**Team:** 
- Priti Sagar (Drexel University)
- Gautam Mahajan (University of Connecticut – Storrs)

📄 Full submission: [SUBMISSION.md](SUBMISSION.md) · [SUBMISSION.pdf](SUBMISSION.pdf)

---

## The Story in Three Validated Layers

1. **Q-MoE Fire** — Regime-diverse quantum experts. California's 58 counties cluster into 4 climate regimes; each is routed through 4 fixed quantum reservoirs with different entanglement topologies (ring, ladder, star, full). **F1 = 0.782, AUC = 0.934** at county level, +10.3% over SVM, with per-regime improvements in all 4 climate zones.
2. **RA-MQTF** — Multi-task fusion on top of Q-MoE expert features jointly predicts fire occurrence, severity, and premiums. Best wildfire classifier (**F1 = 0.834**, +17.6% over SVM) and best severity model (**R2 = 0.639**).
3. **QLSTM** — 4-qubit variational LSTM gates for ZIP-level temporal premium prediction. **R2 = 0.349** on temporal log-scale splits, substantially outperforming Classical LSTM (R2 = −1.033).

**Proposed algorithm:** [ZIP-STR-QMoE](SUBMISSION.md#section-5-envisioned-algorithm--zip-str-qmoe) extends these validated ingredients to direct ZIP-level prediction.

**Main limitation:** Wildfire models operate at county granularity and are mapped to ZIP codes via geographic lookup. ZIP-STR-QMoE addresses this gap directly.

---

## Best Results by Task

| Task | Model | Granularity | Score | Type |
|------|-------|-------------|-------|------|
| 1A: Wildfire classification | **RA-MQTF** (4×4 qubits) | County → ZIP | **F1 = 0.834** | Quantum |
| 1A: Wildfire classification | Q-MoE Fire (4×4 qubits) | County | F1 = 0.782 | Quantum |
| 1A: Classical best | SVM (RBF) | County | F1 = 0.709 | Classical |
| 2: Fire severity | **RA-MQTF** (4×4 qubits) | County | **R2 = 0.639** | Quantum |
| 2: Insurance premium (temporal) | **QLSTM** (4 qubits) | ZIP | **R2 = 0.349** | Quantum |
| 2: Premium (temporal best) | Temporal XGBoost | ZIP | R2 = 0.828 | Classical |
| 2: Premium (non-temporal) | Random Forest | ZIP | R2 = 0.922 | Classical |
| Forward predictions | Q-MoE + ensemble | ZIP | 2,174 ZIPs scored | Forecast |

**Time-series comparison (identical splits):** QLSTM R2 = 0.349 vs Classical LSTM R2 = −1.033 vs Temporal XGBoost R2 = 0.828 (all log-scale temporal). QLSTM substantially outperforms its classical counterpart; XGBoost remains the strongest temporal baseline. Random Forest's R2 = 0.922 is from a non-temporal split (different evaluation setting).

---

## Repository Structure

```
data/                                    # Challenge-provided datasets (wildfire + insurance + GeoJSON)
notebooks/
  01_eda_and_visualization/              # Exploratory analysis & visualization
    01_EDA_and_Baselines.ipynb           # EDA + classical & quantum baselines (Task 1A)
    02_Geospatial_Visualization.ipynb    # Risk maps, scalability analysis
    03_Clustering_UMAP_HDBSCAN.ipynb     # ZIP-code risk clustering into 5 tiers
    04_FullHistory_Wildfire_Analysis.ipynb  # 40-year wildfire trend analysis
  02_quantum_baselines/                  # Quantum ML baselines & architectures
    01_Task2_Insurance_QML.ipynb         # Insurance premium prediction (Task 2)
    02_Improved_Quantum_Models.ipynb     # PCA, data re-uploading, hybrid ensemble
    03_Cloud_Quantum_Models.ipynb        # QRC, QLSTM, Trainable Kernel, Transfer Learning
    04_Advanced_Quantum_Architectures.ipynb # QCNN, PQK, QGAN, Quanvolution, Dressed Circuit
  03_novel_explorations/                 # Research explorations & building blocks
    01_RAM_QRSRE.ipynb                   # RAM-QRSRE — regime-adaptive multi-scale QML
    02_Quantum_Graph_Reservoir.ipynb     # QGR — spatial graph + quantum reservoir
    03_Conformal_Quantum_Prediction.ipynb # CQP — conformal uncertainty quantification
    04_MultiTask_Quantum_Temporal_Fusion.ipynb # MQTF — joint fire + insurance prediction
  04_proposed_algorithm/                 # ★ Competition entry — proposed algorithms
    01_QMoE_Fire.ipynb                   # Q-MoE Fire — regime-diverse experts (F1 = 0.782)
    02_QART.ipynb                        # QART — quantum attention fusion
    03_QCAST_Scenario_Transformer.ipynb  # Q-CAST — scenario analysis
    04_RegimeAware_MQTF.ipynb            # RA-MQTF — best model (F1 = 0.834, R2 = 0.639)
  05_predictions/                        # Final deliverables
    01_ZipCode_2026_Predictions.ipynb    # Per-ZIP wildfire risk predictions for 2,174 ZIPs
results/                                 # Plots, CSVs, JSON outputs
requirements.txt                         # Python dependencies
SUBMISSION.md / SUBMISSION.pdf           # Competition submission document
README.md
```

---

## Quantum Techniques

**Primary competition entries** (`04_proposed_algorithm/`):
- **Q-MoE Fire** — 4×4 qubits, topology-diverse fixed reservoirs (ring/ladder/star/full), regime routing + GradientBoosting head. Best single-task wildfire classifier (F1 = 0.782).
- **RA-MQTF** — Q-MoE expert features + shared multi-task neural backbone. Best overall wildfire classifier (F1 = 0.834) and severity model (R2 = 0.639).
- **QART** — Quantum Adaptive Reservoir Transformer with cross-attention (F1 = 0.755).
- **Q-CAST** — Scenario analysis through the quantum pipeline (6 climate scenarios, conformal 90% intervals).

**Cloud / NISQ-ready models** (`02_quantum_baselines/03_Cloud_Quantum_Models.ipynb`):
- **QLSTM** — 4 qubits, variational LSTM gates with StronglyEntanglingLayers. Temporal premium model (R2 = 0.349, beats Classical LSTM R2 = −1.033). 144 trainable quantum parameters.
- **QRC** — 8 qubits, 60-layer fixed reservoir baseline (F1 = 0.724).
- Trainable Quantum Kernel, Quantum Transfer Learning. AWS Braket integration for IonQ Aria, Rigetti Ankaa-3, IQM Garnet.

**Supporting explorations** (`03_novel_explorations/`):
- **CQP** — Conformal Quantum Prediction with ~90% empirical coverage.
- **MQTF** — Multi-task temporal fusion (+69% F1, +588% R2 over single-task baselines, motivating RA-MQTF).
- **QGR** — Quantum Graph Reservoir using county adjacency (F1 = 0.650).
- **RAM-QRSRE** — Regime-adaptive multi-scale QML.

**Baseline architectures** (`02_quantum_baselines/`):
VQC, VQR, Quantum Kernel SVM, QCNN, PQK, QGAN, Quanvolutional NN, Dressed Quantum Circuit, PCA-reduced quantum encoders, data re-uploading classifiers.

---

## Quantum Resource Requirements

| Model | Qubits | Depth | Trainable Params | Sim. Time |
|-------|--------|-------|------------------|-----------|
| Q-MoE (4 experts) | 4 × 4 | 40 each | 0 fixed + GBM | ~7 min |
| RA-MQTF (multi-task) | 4 × 4 | 40 each | 0 fixed + 6,595 NN | ~6.5 min |
| QLSTM (Task 2) | 4 | 12 | 144 | ~15 s |
| QRC (Task 1A baseline) | 8 | 60 | 0 fixed | ~18 s |

All models run on PennyLane `default.qubit` (exact statevector simulator). Fixed reservoirs use analytic statevector evaluation (equivalent to infinite shots); QLSTM uses parameter-shift gradients (~1.7M circuit evaluations across training).

---

## Quantum Model Progression

| Generation | Notebook | Model | F1 | Insight |
|------------|----------|-------|-----|---------|
| 1 | EDA | VQC | 0.542 | Underperforms classical (barren plateaus) |
| 2 | Cloud QML | QRC | 0.724 | First quantum win over SVM (+2.1%) |
| 3 | Q-MoE Fire | Q-MoE | 0.782 | Regime-aware experts (+10.3% over SVM) |
| 4 | RA-MQTF | RA-MQTF | **0.834** | Multi-task fusion (+17.6% over SVM) |

---

## Setup

```bash
pip install -r requirements.txt
```

All 17 notebooks include pre-executed cell outputs for reproducibility. Run them in folder order: `01_eda_and_visualization/` → `02_quantum_baselines/` → `03_novel_explorations/` → `04_proposed_algorithm/` → `05_predictions/`. Each reads from `data/` and writes to `results/`.

---

## Tech Stack

- **PennyLane** + PyTorch — all primary quantum models (Q-MoE, RA-MQTF, QLSTM, QRC, QART)
- **Qiskit** — VQC, VQR, quantum kernels, advanced architectures
- **AWS Braket** — Cloud quantum hardware integration (IonQ, Rigetti, IQM)
- **scikit-learn, XGBoost** — classical baselines
- **SHAP** — model explainability and feature importance
- **pandas, NumPy, matplotlib, seaborn, Plotly** — data processing and visualization

---

## Data

All analysis uses **only** the challenge-provided files (no external APIs or third-party datasets):
- `wildfire_county_monthly.csv` — monthly county-level wildfire and weather records (2008–2020)
- `wildfire_weather_daily.csv` — daily weather and fire occurrence (used for temporal trend analysis and 2026 projections)
- `insurance_2018_2019.XLS`, `insurance_2020_2021.XLS` — ZIP-level homeowners insurance premiums and risk scores
- `geojson-counties-fips.json` — county boundaries for geospatial visualization

---

## Code Repository

**https://github.com/Priti0427/Quantum-Sustainability-Challenge**

See [SUBMISSION.md](SUBMISSION.md) for the full technical write-up, including the envisioned ZIP-STR-QMoE algorithm (Section 5).
