# Quantum Sustainability Challenge

Quantum Machine Learning solution for the Deloitte Quantum Sustainability Challenge -- predicting California wildfire risk and insurance premiums using hybrid quantum-classical models.

## The Story in Three Layers

1. **Q-MoE Fire** -- regime-diverse quantum experts prove that matching circuit topology to climate structure improves wildfire prediction (F1=0.782, county-level)
2. **RA-MQTF** -- multi-task fusion on top of Q-MoE experts produces the best wildfire classifier (F1=0.834) and severity model (R2=0.639)
3. **QLSTM** -- quantum LSTM gates achieve R2=0.349 on temporal log-scale premium prediction, substantially outperforming Classical LSTM (R2=-1.033)

**Proposed**: ZIP-STR-QMoE extends these validated ingredients to direct ZIP-level prediction (see SUBMISSION.md Section 5)

**Main limitation**: Wildfire models operate at county granularity and are mapped to ZIP codes. The proposed algorithm addresses this gap.

## Tasks and Best Results

| Task | Model | Granularity | Score | Type |
|------|-------|-------------|-------|------|
| 1A: Wildfire classification | RA-MQTF (4x4 qubits) | County | **F1 = 0.834** | Quantum |
| 1A: Wildfire classification | Q-MoE Fire (4x4 qubits) | County | F1 = 0.782 | Quantum |
| 1A: Classical best | SVM (RBF) | Daily | F1 = 0.709 | Classical |
| 2: Fire severity | RA-MQTF (4x4 qubits) | County | **R2 = 0.639** | Quantum |
| 2: Insurance premium | QLSTM (4 qubits) | ZIP | **R2 = 0.349** (log-scale temporal) | Quantum |
| 2: Premium (temporal best) | Temporal XGBoost | ZIP | R2 = 0.828 | Classical |
| 2: Premium (non-temporal) | Random Forest | ZIP | R2 = 0.922 | Classical |
| Forward predictions | Q-MoE + ensemble | ZIP | 2,174 ZIPs scored | Forecast |

**Time-series comparison (identical splits):** QLSTM R2=0.349 vs Classical LSTM R2=-1.033 vs Temporal XGBoost R2=0.828 (all log-scale temporal). QLSTM substantially outperforms Classical LSTM; Temporal XGBoost is the strongest temporal baseline. Random Forest achieves R2=0.922 on a non-temporal split (different evaluation setting).

## Project Structure

```
data/                                  # Datasets (wildfire + insurance)
notebooks/
  01_eda_and_visualization/            # Exploratory analysis & visualization
    01_EDA_and_Baselines.ipynb             # EDA + classical & quantum baselines (Task 1A)
    02_Geospatial_Visualization.ipynb      # Maps, scalability analysis
    03_Clustering_UMAP_HDBSCAN.ipynb       # ZIP-code risk clustering (5 tiers)
    04_FullHistory_Wildfire_Analysis.ipynb  # Extended 40-year wildfire analysis
  02_quantum_baselines/                # Quantum ML baselines & architectures
    01_Task2_Insurance_QML.ipynb           # Insurance premium prediction (Task 2)
    02_Improved_Quantum_Models.ipynb       # PCA, data re-uploading, hybrid ensemble
    03_Cloud_Quantum_Models.ipynb          # QRC, QLSTM, Trainable Kernel, Transfer Learning
    04_Advanced_Quantum_Architectures.ipynb # QCNN, PQK, QGAN, Quanvolution, Dressed Circuit
  03_novel_explorations/               # Research explorations & building blocks
    01_RAM_QRSRE.ipynb                     # RAM-QRSRE -- regime-adaptive multi-scale QML
    02_Quantum_Graph_Reservoir.ipynb       # QGR -- spatial graph + quantum reservoir
    03_Conformal_Quantum_Prediction.ipynb  # CQP -- conformal uncertainty quantification
    04_MultiTask_Quantum_Temporal_Fusion.ipynb # MQTF -- joint fire + insurance prediction
  04_proposed_algorithm/               # Competition entry -- proposed algorithms
    01_QMoE_Fire.ipynb                     # Q-MoE Fire -- regime-diverse experts (F1=0.782)
    02_QART.ipynb                          # QART -- quantum attention fusion
    03_QCAST_Scenario_Transformer.ipynb    # Q-CAST -- scenario analysis
    04_RegimeAware_MQTF.ipynb              # RA-MQTF -- best model (F1=0.834, R2=0.639)
  05_predictions/                      # Final deliverables
    01_ZipCode_2026_Predictions.ipynb      # Per-ZIP wildfire risk predictions
results/                               # Plots, CSVs, JSON outputs
requirements.txt                       # Python dependencies
SUBMISSION.md                          # Competition submission document
README.md
```

## Quantum Techniques

**Primary (validated, competition-relevant):**
- **Q-MoE Fire** -- 4x4 qubits, topology-diverse fixed reservoirs, regime routing. Best single-task wildfire classifier (F1=0.782)
- **RA-MQTF** -- Q-MoE experts + multi-task backbone. Best overall wildfire classifier (F1=0.834) and severity model (R2=0.639)
- **QLSTM** -- 4 qubits, variational LSTM gates. Temporal premium model (R2=0.349, beats Classical LSTM R2=-1.033)
- **QRC** -- 8 qubits, 15 fixed layers. Baseline quantum classifier (F1=0.724)

**Supporting explorations** (documented in `03_novel_explorations/`):
- QART: multi-scale quantum reservoirs + cross-attention (F1=0.755)
- QGR: county adjacency graph + quantum reservoir (F1=0.650)
- CQP: conformal uncertainty quantification (~90% coverage)
- MQTF: multi-task temporal fusion (F1=0.693)
- Q-CAST: scenario analysis (6 climate scenarios, decision support)
- RAM-QRSRE: regime-adaptive multi-scale QML

**Baseline architectures** (documented in `02_quantum_baselines/`):
- VQC, VQR, Quantum Kernel SVM, QCNN, PQK, QGAN, Quanvolutional NN, Dressed Quantum Circuit, Trainable Kernel, Quantum Transfer Learning

## Quantum Resource Requirements

| Model | Qubits | Depth | Trainable Params | Time |
|-------|--------|-------|-------------------|------|
| Q-MoE (4 experts) | 4 x 4 | 40 each | 0 (fixed) + GBM | ~7 min |
| RA-MQTF (multi-task) | 4 x 4 | 40 each | 0 (fixed) + 6,595 NN | ~6.5 min |
| QLSTM | 4 | 12 | 144 | 15s |
| QRC | 8 | 60 | 0 (fixed) | 18s |

## Setup

```bash
pip install -r requirements.txt
```

## Tech Stack

- **PennyLane** + PyTorch (all quantum models)
- **Qiskit** (VQC, VQR, quantum kernels)
- **scikit-learn, XGBoost** (classical baselines)
- **SHAP** (model explainability)
- **pandas, matplotlib, seaborn, Plotly** (visualization)

## End-to-End Pipeline

Run the notebooks in order. Each reads from `data/` and writes to `results/`.

```
01_eda_and_visualization/
  01 -> 02 -> 03 -> 04    Baselines, maps, clustering, history

02_quantum_baselines/
  01 -> 02 -> 03 -> 04    Task 2 baselines, improved models, QRC/QLSTM, advanced architectures

03_novel_explorations/
  01 -> 02 -> 03 -> 04    RAM-QRSRE, QGR, CQP, MQTF (building blocks)

04_proposed_algorithm/     Competition entry
  01 Q-MoE Fire            Regime-diverse experts (Task 1A)
  02 QART                  Quantum attention fusion
  03 Q-CAST                Scenario analysis
  04 RA-MQTF               Multi-task fusion (best model)

05_predictions/
  01 ZipCode Predictions   Final per-ZIP wildfire risk predictions
```

All notebooks are committed with full cell outputs.
