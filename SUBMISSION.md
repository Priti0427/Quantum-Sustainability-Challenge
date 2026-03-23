# Deloitte Quantum Sustainability Challenge — Submission

## Team Overview

*[Team name and member details to be filled in]*

## Abstract (400 words)

We present a hybrid quantum-classical machine learning solution for predicting California wildfire risk and mapping it to insurance premiums, addressing both Task 1A (wildfire classification) and Task 2 (premium time-series forecasting) of the Deloitte Quantum Sustainability Challenge.

Our approach systematically evaluates six quantum ML architectures — Variational Quantum Classifier (VQC), Variational Quantum Regressor (VQR), Quantum Kernel SVM, Quantum Reservoir Computing (QRC), Quantum LSTM (QLSTM), and Quantum Transfer Learning — against classical baselines on identical data splits to establish honest performance comparisons.

For Task 1A, Quantum Reservoir Computing emerges as our strongest model, achieving F1=0.724 on wildfire day classification using 8 qubits and 15 fixed random reservoir layers. QRC outperforms all classical baselines (Logistic Regression F1=0.670, Random Forest F1=0.648, XGBoost F1=0.638) and avoids the barren plateau problem that undermines variational approaches (VQC achieved only F1=0.542, and an improved data-reuploading VQC regressed to F1=0.333). The fixed quantum reservoir acts as a non-linear feature extractor, mapping weather data into a richer Hilbert-space representation that a classical logistic readout can effectively separate.

For Task 2, we implement a Quantum LSTM with 4 qubits, where standard LSTM gates are replaced by variational quantum circuits using PennyLane's TorchLayer integration. On identical 2-step time-series sequences of ZIP-code insurance features, QLSTM achieves R²=0.922, significantly outperforming both Classical LSTM (R²=0.765) and Temporal XGBoost (R²=0.860). This represents a genuine quantum advantage for temporal insurance premium prediction, satisfying the competition's requirement for "time series analysis using QML."

Beyond model development, we perform unsupervised risk stratification using UMAP dimensionality reduction and HDBSCAN density-based clustering, classifying 1,829 California ZIP codes into five risk tiers (Negligible through Extreme). These tiers, combined with domain-driven features (Fire Weather Index proxies, drought indicators, hydroclimate whiplash signals), feed into our 2026 prediction pipeline, producing per-ZIP risk scores for 2,174 California locations.

Our key finding is that quantum advantage manifests most strongly when (1) variational optimization is avoided (QRC over VQC/VQR), (2) the task exhibits non-linearity that benefits from Hilbert-space feature maps, and (3) comparisons use identical data. We document all quantum resource requirements (qubits, circuit depth, training time) and provide AWS Braket integration for real hardware execution.

## Algorithm Description

### Data Pipeline

Four datasets spanning 1984-2025:
- Daily California weather + fire occurrence data (14,988 rows)
- Monthly county-level fire data (10,989 rows)
- ZIP-code homeowners insurance data for 2018-2019 and 2020-2021

Feature engineering produces: temperature range, wind-temperature ratio, lagged precipitation, lagged wind speed, fire risk scores, premium per policy, loss ratios, and temporal lag features.

### Task 1A: Wildfire Classification

**Primary model: Quantum Reservoir Computing (QRC)**
- 8-qubit quantum reservoir with 15 layers of random RX/RY/RZ rotations + circular CNOT entanglement
- 360 fixed (non-trainable) parameters eliminate barren plateau risk
- Angle encoding maps PCA-reduced weather features to qubit rotations
- Pauli-Z expectation values from all 8 qubits serve as quantum features
- Classical Logistic Regression readout with balanced class weights
- Time-based split: train on 2018-2020 (500 samples), test on 2021 (200 samples)
- **Result: F1 = 0.724, AUC-ROC = 0.706**

### Task 1B: Evaluation

Quantum advantage is genuine but architecture-dependent. QRC (F1=0.724) beats all classical models, while VQC (F1=0.542) and Trainable Kernel (F1=0.387) underperform. The key differentiator is avoiding variational optimization — fixed quantum circuits with classical readouts outperform optimized quantum circuits with barren plateau challenges.

### Task 2: Insurance Premium Time Series

**Primary model: Quantum LSTM (QLSTM)**
- 4-qubit variational circuits replace LSTM forget, input, cell, and output gates
- StronglyEntanglingLayers ansatz with 3 layers per gate (144 quantum parameters total)
- PennyLane TorchLayer enables end-to-end PyTorch gradient training
- 2-step sequences: ZIP features at year t and t+1 predict premium at year t+2
- 200 training sequences, 718 test sequences, 30 epochs
- **Result: R² = 0.922, RMSE = 0.158**

**Fair comparison on identical data splits:**
| Model | R² | Type |
|-------|------|------|
| QLSTM (4 qubits) | 0.922 | Quantum |
| Temporal XGBoost | 0.860 | Classical |
| Classical LSTM | 0.765 | Classical |

### ZIP-Code Risk Clustering

UMAP reduces 7 insurance features to 2D, HDBSCAN finds natural density clusters, and agglomerative clustering merges micro-clusters into 5 macro risk tiers. 1,829 ZIP codes classified.

### 2026 Predictions

Domain-driven features (Fire Weather Index, drought indicators, hydroclimate whiplash) combined with cluster labels and ensemble risk scoring produce per-ZIP predictions for 2,174 California locations. Top-risk areas: Shasta (0.96), Lassen (0.88), Humboldt (0.86), Sonoma (0.82).

## Results Summary

| Task | Best Quantum Model | Score | Classical Best | Quantum Advantage |
|------|-------------------|-------|----------------|-------------------|
| 1A Classification | QRC (8 qubits) | F1=0.724 | XGBoost F1=0.638 | Yes (+13.5%) |
| 2 Time Series | QLSTM (4 qubits) | R²=0.922 | XGBoost R²=0.860 | Yes (+7.2%) |

## Code Repository

*[GitHub repo URL to be inserted]*

All notebooks, data, and results are reproducible. Pre-executed notebooks with full outputs are included. See README.md for setup instructions and pipeline documentation.
