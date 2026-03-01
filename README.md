# Quantum Sustainability Challenge

Quantum Machine Learning solution for the Deloitte Quantum Sustainability Challenge — predicting California wildfire risk and insurance premiums using hybrid quantum-classical models.

## Project Structure

```
├── data/                          # Datasets (wildfire + insurance)
├── notebooks/
│   ├── 01_EDA_and_Baselines.ipynb       # EDA + classical & quantum baselines (Task 1A)
│   ├── 02_Task2_Insurance_QML.ipynb     # Insurance premium prediction (Task 2)
│   ├── 03_Geospatial_Visualization.ipynb # Maps, scalability analysis
│   └── 04_Improved_Quantum_Models.ipynb  # PCA, data re-uploading, hybrid ensemble
├── results/                       # Plots, CSVs, and JSON outputs
├── requirements.txt               # Python dependencies
└── README.md
```

## Tasks

| Task | Description | Best Model | Score |
|------|-------------|-----------|-------|
| 1A | Wildfire day prediction (binary classification) | Hybrid Ensemble (XGB+RF+LR+VQC) | F1 = 0.705 |
| 2 | Insurance premium forecasting (regression) | Hybrid Ensemble (XGB+RF+LinReg+VQR) | R² = 0.998 |

## Quantum Techniques Used

- **Variational Quantum Classifier (VQC)** with data re-uploading
- **Variational Quantum Regressor (VQR)** with log-transformed targets
- **Quantum Kernel SVM / Ridge Regression**
- **PCA preprocessing** for decorrelated quantum encoding
- **Hybrid quantum-classical stacking ensembles**

## Setup

```bash
pip install -r requirements.txt
```

## Tech Stack

- **Qiskit** (quantum circuits, VQC, VQR, quantum kernels)
- **scikit-learn, XGBoost** (classical baselines, ensembles)
- **pandas, matplotlib, seaborn** (data analysis, visualization)
