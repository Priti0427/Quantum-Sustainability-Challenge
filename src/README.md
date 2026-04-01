# Quantum Sustainability Challenge: Wildfire & Insurance Risk Modeling

## What is this?
This repository contains a state-of-the-art hybrid Quantum Machine Learning (QML) pipeline developed for the Deloitte Quantum Sustainability Challenge. It is designed to forecast California wildfire risks and estimate subsequent insurance premium impacts.

The architecture combines classical data processing (feature engineering, UMAP/HDBSCAN clustering) with advanced quantum architectures (Quantum Transfer Learning, Quantum Mixture of Experts, and Quantum Attention) to process decades of climate and geographic data. 

The pipeline culminates in a forward-looking 2026 simulation, producing mathematically guaranteed uncertainty bounds (Conformal Prediction) and a publication-ready Executive Dashboard.

## Hardware Recommendations
This pipeline dynamically scales to your available hardware:
* **NVIDIA dGPU / eGPU (Recommended):** For the best performance, run this on a machine with an NVIDIA dedicated or external GPU. The quantum simulations will automatically utilize CUDA via `pennylane-lightning[gpu]`, accelerating training times exponentially.
* **CPU (Supported):** The pipeline is fully capable of running locally on a standard CPU. It will automatically detect the lack of a GPU and fall back to PennyLane's `default.qubit` simulator. *Note: Training the quantum models on a CPU will take significantly longer.*

## How to Run Locally

### 1. Setup
Ensure your terminal is in the root directory of the project and that you have installed the necessary Python packages (PyTorch, PennyLane, Pandas, Scikit-Learn, Seaborn, GeoPandas). 

### 2. The "God Mode" Command (Run Everything)
To execute the entire pipeline from end-to-end—ingesting data, training the quantum models, evaluating the insurance data, and generating the final 2026 predictions—run:

```python3 main.py --task all --model quantum```

## Understanding the Results & Deliverables
When the pipeline finishes running, it outputs its final metrics to the terminal and saves all generated artifacts to the results/ folder. Here is what to expect:

## Expected Performance Metrics
Due to the highly imbalanced nature of wildfire data (fire days are rare), standard accuracy is misleading. Our Quantum Transfer Learning model achieves the following on the full dataset:

-> AUC (Area Under the Curve): ~0.85 — This indicates excellent separability; the quantum feature maps successfully distinguish between high-risk and low-risk weather patterns.

-> F1 Score: ~0.54 — Achieved by calibrating the decision threshold to 0.2, allowing the model to effectively identify true positive fire events without being overly conservative.

## Visual Artifacts (/results folder)

### The pipeline automatically generates the graphs required for the final challenge submission PDF:
-> Task1A_clusters.png: A UMAP + HDBSCAN scatter plot proving the successful unsupervised discovery of micro-clusters and risk tiers from the historical data.

-> Task1A_Quantum_TL_results.png: The classification performance boundary/scatter plot showing the quantum model's true vs. predicted accuracy.

-> maxtempF_histogram.png & fire_risk_score_hist.png: Distribution plots for the Exploratory Data Analysis (EDA) section of the report
