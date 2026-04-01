import argparse
import os
import numpy as np
import pandas as pd

# Configuration & Data
from src import config
from src.loader import load_csv
from src.wildfire import prepare_wildfire
from src.feature import create_features
from src.preprocessing import split_scale
from src.clustering import run_clustering

# Visualizations
from src.visualization import plot_clusters, plot_hist, plot_model_results
from src.geospatial import plot_map

# Models
from src.classical import train_rf, evaluate
from src.gpu import to_numpy
from src.quantum import run_transfer_learning
from src.insurance_modeling import load_insurance

# Final Predictions
from src.predict_2026 import run_2026_pipeline

def main():
    parser = argparse.ArgumentParser(description="Quantum Sustainability Challenge Pipeline")
    # Added 'all' to the choices here!
    parser.add_argument('--task', type=str, choices=['1A', '2', 'predict2026', 'all'], required=True, help="Task to run (use 'all' to run everything)")
    parser.add_argument('--model', type=str, choices=['classical', 'quantum'], default='classical', help="Model architecture")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Notice these are now all standalone 'if' statements checking for 'all'
    if args.task in ['1A', 'all']:
        print("\n=== [ TASK 1A: Wildfire Classification ] ===")
        print("[1/4] Loading and Preprocessing Wildfire Data...")
        df = load_csv(config.DATA_PATHS["wildfire"])
        df = prepare_wildfire(df)
        df = create_features(df)

        print("[2/4] Generating EDA Visualizations...")
        plot_hist(df, "maxtempF")
        print("      -> Saved: results/maxtempF_histogram.png")

        X_train, X_test, y_train, y_test = split_scale(df, config.TARGET_COL, config.TEST_SIZE, config.RANDOM_STATE)

        print("[3/4] Running UMAP + HDBSCAN Clustering...")
        emb, labels = run_clustering(X_train)
        plot_clusters(emb, labels, filename="results/task1A_clusters.png")
        print("      -> Saved: results/task1A_clusters.png")

        print(f"[4/4] Training {args.model.capitalize()} Model...")
        if args.model == 'classical':
            model = train_rf(X_train, y_train)
            evaluate(model, X_test, y_test)
            
            # Generate Performance Graph
            preds = to_numpy(model.predict(X_test))
            res_img = plot_model_results(to_numpy(y_test), preds, "Task1A", "Classical_RF")
            print(f"      -> Saved Performance Graph: {res_img}")

        elif args.model == 'quantum':
            # Fast test params (increase n_train for real runs)
            f1, auc, preds, probs = run_transfer_learning(X_train, X_test, y_train, y_test, n_train=len(X_train), epochs=30)
            print(f"      -> Quantum Transfer Learning | F1: {f1:.4f} | AUC: {auc:.4f}")
            
            res_img = plot_model_results(to_numpy(y_test), preds, "Task1A", "Quantum_TL")
            print(f"      -> Saved Performance Graph: {res_img}")

    if args.task in ['2', 'all']:
        print("\n=== [ TASK 2: Insurance Premium Regression ] ===")
        print("[1/3] Loading Insurance Data (2018-2021)...")
        model, ins_df, col_prem, col_exp, col_risk = load_insurance(
            config.DATA_PATHS["insurance_2018"], 
            config.DATA_PATHS["insurance_2020"]
        )
        
        print("[2/3] Extracting Features and Plotting Distributions...")
        plot_hist(ins_df, col_risk, filename="results/fire_risk_score_hist.png")
        print("      -> Saved: results/fire_risk_score_hist.png")

        print("[3/3] Task 2 Baseline Training Complete.")

    if args.task in ['predict2026', 'all']:
        print("\n=== [ FULL PIPELINE: 2026 Final Predictions ] ===")
        
        print("[1/3] Loading Actual Insurance & Geographic Data...")
        model, ins_df, col_prem, col_exp, col_risk = load_insurance(
            config.DATA_PATHS["insurance_2018"], 
            config.DATA_PATHS["insurance_2020"]
        )
        
        print(f"[2/3] Extracting {len(ins_df['ZIP_Code'].unique())} real California ZIP Codes...")
        # Get the most recent data (2021) to act as the baseline for 2026 predictions
        latest_data = ins_df[ins_df['Year'] == 2021].copy()
        
        print("[3/3] Generating Conformal Prediction Intervals & ZIP Forecasts...")
        # Pass the ACTUAL dataframe and your trained model into the finale
        run_2026_pipeline(latest_data, config.DATA_PATHS["geo"])
        
        print("\nPipeline Complete. Check the /results/ folder for your final PDF deliverables.")

if __name__ == "__main__":
    main()
