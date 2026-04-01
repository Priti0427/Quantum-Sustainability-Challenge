import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

def plot_clusters(emb, labels, filename="results/clusters.png"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='Spectral', s=15, alpha=0.7)
    plt.colorbar(scatter, label="HDBSCAN Cluster ID")
    plt.title("UMAP Dimensionality Reduction & HDBSCAN Clusters")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_hist(df, col, filename=None):
    if filename is None:
        filename = f"results/{col}_histogram.png"
    plt.figure(figsize=(6, 4))
    df[col].hist(bins=30, color='coral', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_model_results(y_true, y_pred, task_name, model_type):
    """Plots True vs Predicted values or Probability distributions."""
    filename = f"results/{task_name}_{model_type}_results.png"
    plt.figure(figsize=(8, 6))
    
    # Check if regression (continuous) or classification (binary)
    if len(np.unique(y_true)) > 2:
        plt.scatter(y_true, y_pred, alpha=0.5, color='dodgerblue')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_type.capitalize()} Model Performance ({task_name})")
    else:
        # Classification: Plot histograms of probabilities or scatter of classes
        plt.scatter(range(len(y_true)), y_true, label="Actual", alpha=0.6, marker='o')
        plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.6, marker='x')
        plt.xlabel("Sample Index")
        plt.ylabel("Class (0=No Fire, 1=Fire)")
        plt.title(f"{model_type.capitalize()} Predictions ({task_name})")
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename
