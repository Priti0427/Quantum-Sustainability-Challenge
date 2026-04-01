"""
insurance_modeling.py — Task 2 data preparation and Conformal Prediction.

Includes:
- Loading and merging 2018-2021 insurance datasets
- Building time-series sequences for QLSTM
- NEW: Conformal Prediction wrappers for mathematically guaranteed uncertainty bounds
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from .gpu import LinearRegression, to_device, to_numpy

SEED = 42

# ─── 1. DATA LOADING & SEQUENCING ─────────────────────────────────────────────

def _find_col(df, keywords):
    """Find a column whose name contains all keywords (case-insensitive)."""
    kw = [k.lower() for k in keywords]
    for c in df.columns:
        if all(k in c.lower() for k in kw):
            return c
    return None


def _load_ho(path, years):
    """Load homeowners insurance XLS, one sheet per year (e.g. '2018HO')."""
    frames = []
    for yr in years:
        df = pd.read_excel(path, sheet_name=f"{yr}HO", header=1, engine="openpyxl")
        df.columns = [c.replace("\n", " ").strip() for c in df.columns]
        df = df.rename(columns={df.columns[0]: "ZIP_Code"})
        df["ZIP_Code"] = pd.to_numeric(df["ZIP_Code"], errors="coerce")
        df = df.dropna(subset=["ZIP_Code"])
        df["ZIP_Code"] = df["ZIP_Code"].astype(int)
        df["Year"] = yr
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_insurance(path_2018, path_2020):
    """
    Load and merge both XLS files, engineer features like loss ratio,
    and fit a baseline classical Linear Regression on earned premium.
    Also returns the merged DataFrame and key column names for downstream use.
    """
    ins1 = _load_ho(path_2018, [2018, 2019])
    ins2 = _load_ho(path_2020, [2020, 2021])
    common = list(set(ins1.columns) & set(ins2.columns))
    ins = pd.concat([ins1[common], ins2[common]], ignore_index=True)

    col_prem = _find_col(ins, ["earned", "premium"])
    col_exp  = _find_col(ins, ["earned", "exposure"])
    col_risk = _find_col(ins, ["fire", "risk", "score"])
    
    loss_cols = [c for c in ins.columns if "fire" in c.lower() and "incurred" in c.lower()]
    ins["total_insured_loss"] = ins[loss_cols].fillna(0).sum(axis=1)

    feature_cols = [col_risk, col_exp, "total_insured_loss"]
    df = ins.dropna(subset=feature_cols + [col_prem]).copy()

    X = df[feature_cols].values
    y = df[col_prem].values

    model = LinearRegression()
    model.fit(to_device(X), to_device(y))

    print(f"Classical Insurance baseline trained on {len(df)} rows")
    return model, ins, col_prem, col_exp, col_risk


def build_insurance_sequences(ins, col_prem, feature_cols):
    """
    Build 2-step ZIP-code yearly sequences for QLSTM regression.
    Returns sequences (N, 2, n_features) float32, targets (N,) float32.
    """
    ins_ts = ins.dropna(subset=feature_cols + ["Year"]).copy()
    ins_ts = ins_ts.sort_values(["ZIP_Code", "Year"])

    sc = StandardScaler()
    ins_ts[feature_cols] = sc.fit_transform(ins_ts[feature_cols])

    sequences, targets = [], []
    for _, grp in ins_ts.groupby("ZIP_Code"):
        grp = grp.sort_values("Year")
        if len(grp) < 3:
            continue
        feats = grp[feature_cols].values
        prems = grp[col_prem].values
        for i in range(len(grp) - 2):
            seq = feats[i:i+2]
            tgt = np.log1p(np.abs(prems[i+2])) * np.sign(prems[i+2])
            sequences.append(seq)
            targets.append(tgt)

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


# ─── 2. CONFORMAL QUANTUM PREDICTION ──────────────────────────────────────────

class ConformalWrapper:
    """
    Wraps any trained ML/Quantum model to provide statistically guaranteed 
    uncertainty bounds using Split Conformal Prediction.
    """
    def __init__(self, base_model, task_type="regression", alpha=0.10):
        """
        Args:
            base_model: A trained model with a .predict() or .predict_proba() method (or PyTorch equivalent)
            task_type: "regression" (Task 2) or "classification" (Task 1A)
            alpha: Target error rate (e.g., 0.10 means 90% confidence intervals)
        """
        self.base_model = base_model
        self.task_type = task_type
        self.alpha = alpha
        self.q_hat = None

    def _get_predictions(self, X):
        """Helper to extract predictions whether it's Sklearn or PyTorch."""
        import torch
        if isinstance(self.base_model, torch.nn.Module):
            self.base_model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                preds = self.base_model(X_tensor).cpu().numpy()
                return preds.flatten() if self.task_type == "regression" else preds
        else:
            if self.task_type == "classification":
                return self.base_model.predict_proba(X)[:, 1] # Get prob of class 1
            return self.base_model.predict(X)

    def calibrate(self, X_cal, y_cal):
        """
        Calculate non-conformity scores on a hold-out calibration set.
        """
        preds = self._get_predictions(X_cal)
        n = len(y_cal)
        
        if self.task_type == "regression":
            # Absolute error as non-conformity score
            scores = np.abs(y_cal - preds)
        else:
            # 1 - Probability of the true class
            scores = np.where(y_cal == 1, 1 - preds, preds)

        # Calculate the required empirical quantile
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0) # bound between 0 and 1
        self.q_hat = np.quantile(scores, q_level, method='higher')
        
        print(f"Conformal Calibration Complete. Task: {self.task_type.capitalize()}, "
              f"Target Coverage: {(1-self.alpha)*100:.0f}%, q_hat threshold: {self.q_hat:.4f}")

    def predict_with_bounds(self, X_test):
        """
        Generate predictions with mathematically guaranteed intervals/sets.
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate(X_cal, y_cal) before predicting bounds.")
            
        preds = self._get_predictions(X_test)
        
        if self.task_type == "regression":
            lower_bound = preds - self.q_hat
            upper_bound = preds + self.q_hat
            return preds, lower_bound, upper_bound
        else:
            # For classification, return prediction probabilities and the valid prediction sets
            # 1 means 'Fire', 0 means 'No Fire'. A set can contain both {0, 1} if uncertain.
            pred_sets = []
            for p in preds:
                valid_classes = []
                if p >= 1 - self.q_hat:     # Prob of class 1 is high enough
                    valid_classes.append(1)
                if (1 - p) >= 1 - self.q_hat: # Prob of class 0 is high enough
                    valid_classes.append(0)
                pred_sets.append(valid_classes)
            return preds, pred_sets
