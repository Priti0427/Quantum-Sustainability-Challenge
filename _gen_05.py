"""Generate Notebook 05 — Cloud-Ready Quantum Models (QRC, Kernels, QLSTM, Transfer Learning)."""
import json, os

cells = []
def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})

# ═══════════════════════════════════════════════════════════════════════════════
md("""\
# Notebook 05 — Cloud-Ready Advanced Quantum Models

## Overview
This notebook implements **four bold quantum ML experiments** designed to run both locally
and on **AWS Braket** (GPU simulators or real quantum hardware). Flip `CLOUD_MODE = True`
when running on AWS.

| Section | Technique | Task | Why It Matters |
|---------|-----------|------|----------------|
| **A** | Quantum Reservoir Computing (QRC) | 1A + 2 | Novel — avoids barren plateaus entirely |
| **B** | Trainable Quantum Kernel | 1A | Best standalone quantum results so far |
| **C** | Quantum LSTM (QLSTM) | 2 | Proper time-series QML (challenge requirement) |
| **D** | Quantum Transfer Learning | 1A + 2 | Practical hybrid that leverages both paradigms |
| **E** | Real Hardware (optional) | best model | Massive differentiator vs other submissions |

## Technical Choices
- **PennyLane** + PyTorch for gradient-based training (Adam, not COBYLA)
- `default.qubit` locally → `braket.local.qubit` or `braket.aws.qubit` on AWS
- All models use **parameter-shift** or **backprop** differentiation
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 0 — Setup & Cloud Configuration")
code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time, json as _json
warnings.filterwarnings('ignore')

import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                             mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 150})
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CLOUD_MODE: Set True when running on AWS Braket                           ║
# ║  Set False for local laptop execution                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
CLOUD_MODE = False

N_QUBITS = 8

if CLOUD_MODE:
    try:
        dev_name = 'braket.local.qubit'
        dev = qml.device(dev_name, wires=N_QUBITS)
        print(f'AWS Braket device: {dev_name}  |  {N_QUBITS} qubits')
    except Exception:
        dev = qml.device('default.qubit', wires=N_QUBITS)
        print(f'Braket unavailable, falling back to default.qubit  |  {N_QUBITS} qubits')
else:
    dev = qml.device('default.qubit', wires=N_QUBITS)
    print(f'Local device: default.qubit  |  {N_QUBITS} qubits')

print(f'PennyLane {qml.__version__}  |  PyTorch {torch.__version__}')
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 1 — Load Data for Both Tasks")
code("""\
# ── Task 1A: Wildfire classification ─────────────────────────────────────────
wf = pd.read_csv('../data/wildfire_weather_daily.csv')
wf['DATE'] = pd.to_datetime(wf['DATE'], errors='coerce')
wf = wf[(wf['DATE'].dt.year >= 2018) & (wf['DATE'].dt.year <= 2021)].copy()

TASK1_FEATURES = ['PRECIPITATION', 'MAX_TEMP', 'MIN_TEMP', 'AVG_WIND_SPEED',
                  'TEMP_RANGE', 'WIND_TEMP_RATIO', 'MONTH', 'DAY_OF_YEAR',
                  'LAGGED_PRECIPITATION', 'LAGGED_AVG_WIND_SPEED']
wf = wf.dropna(subset=TASK1_FEATURES + ['FIRE_START_DAY'])

X1 = wf[TASK1_FEATURES].values
y1 = wf['FIRE_START_DAY'].astype(int).values
years = wf['DATE'].dt.year.values

# Time-based split: 2018-2020 train, 2021 test
train1 = years < 2021
X1_tr, X1_te = X1[train1], X1[~train1]
y1_tr, y1_te = y1[train1], y1[~train1]

sc1 = StandardScaler()
X1_tr_s = sc1.fit_transform(X1_tr)
X1_te_s = sc1.transform(X1_te)

# PCA to N_QUBITS dimensions for quantum encoding
pca1 = PCA(n_components=N_QUBITS)
X1_tr_q = pca1.fit_transform(X1_tr_s)
X1_te_q = pca1.transform(X1_te_s)

# Scale to [0, pi] for angle encoding
mm1 = MinMaxScaler(feature_range=(0, np.pi))
X1_tr_q = mm1.fit_transform(X1_tr_q)
X1_te_q = mm1.transform(X1_te_q)

print(f'Task 1A — Train: {X1_tr_q.shape}, Test: {X1_te_q.shape}')
print(f'  PCA variance explained: {pca1.explained_variance_ratio_.sum():.1%}')
print(f'  Fire days — train: {y1_tr.sum()}/{len(y1_tr)}, test: {y1_te.sum()}/{len(y1_te)}')

# ── Task 2: Insurance premium regression ─────────────────────────────────────
def find_col(df, keywords):
    kw = [k.lower() for k in keywords]
    for c in df.columns:
        if all(k in c.lower() for k in kw):
            return c
    return None

def load_ho(path, years):
    frames = []
    for yr in years:
        df = pd.read_excel(path, sheet_name=f'{yr}HO', header=1, engine='openpyxl')
        df.columns = [c.replace('\\n', ' ').strip() for c in df.columns]
        df = df.rename(columns={df.columns[0]: 'ZIP_Code'})
        df['ZIP_Code'] = pd.to_numeric(df['ZIP_Code'], errors='coerce')
        df = df.dropna(subset=['ZIP_Code'])
        df['ZIP_Code'] = df['ZIP_Code'].astype(int)
        df['Year'] = yr
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

ins1 = load_ho('../data/insurance_2018_2019.XLS', [2018, 2019])
ins2 = load_ho('../data/insurance_2020_2021.XLS', [2020, 2021])
common = list(set(ins1.columns) & set(ins2.columns))
ins = pd.concat([ins1[common], ins2[common]], ignore_index=True)

col_prem = find_col(ins, ['earned', 'premium'])
col_exp  = find_col(ins, ['earned', 'exposure'])
col_risk = find_col(ins, ['fire', 'risk', 'score'])
loss_cols = [c for c in ins.columns if 'fire' in c.lower() and 'incurred' in c.lower()]
smoke_cols = [c for c in ins.columns if 'smoke' in c.lower() and 'incurred' in c.lower()]
ins['total_fire_loss'] = ins[loss_cols].fillna(0).sum(axis=1)
ins['total_smoke_loss'] = ins[smoke_cols].fillna(0).sum(axis=1)
ins['total_insured_loss'] = ins['total_fire_loss'] + ins['total_smoke_loss']
ins['prem_per_pol'] = ins[col_prem] / ins[col_exp].replace(0, np.nan)
ins['loss_ratio'] = ins['total_insured_loss'] / ins[col_prem].replace(0, np.nan)

T2_FEATURES = [col_risk, col_exp, 'total_insured_loss']
ins2_clean = ins.dropna(subset=T2_FEATURES + ['Year']).copy()

# Create next-year premium as target
ins2_clean = ins2_clean.sort_values(['ZIP_Code', 'Year'])
ins2_clean['next_prem'] = ins2_clean.groupby('ZIP_Code')[col_prem].shift(-1)
ins2_clean = ins2_clean.dropna(subset=['next_prem'])

X2 = ins2_clean[T2_FEATURES].values
y2_raw = ins2_clean['next_prem'].values
y2 = np.log1p(np.abs(y2_raw)) * np.sign(y2_raw)  # log-transform

# 80/20 split
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=SEED)

sc2 = StandardScaler()
X2_tr_s = sc2.fit_transform(X2_tr)
X2_te_s = sc2.transform(X2_te)

pca2 = PCA(n_components=min(N_QUBITS, X2_tr_s.shape[1]))
X2_tr_q = pca2.fit_transform(X2_tr_s)
X2_te_q = pca2.transform(X2_te_s)

mm2 = MinMaxScaler(feature_range=(0, np.pi))
X2_tr_q = mm2.fit_transform(X2_tr_q)
X2_te_q = mm2.transform(X2_te_q)

print(f'\\nTask 2 — Train: {X2_tr_q.shape}, Test: {X2_te_q.shape}')
print(f'  PCA variance: {pca2.explained_variance_ratio_.sum():.1%}')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: QRC
# ═══════════════════════════════════════════════════════════════════════════════
md("""\
---
# Section A — Quantum Reservoir Computing (QRC)

**Why QRC?**
- Uses a **fixed** random quantum circuit as a feature extractor — no variational optimization
- Completely avoids the **barren plateau** problem that cripples VQC/VQR
- Only a classical readout layer is trained (Ridge regression) — trains in seconds
- Competitive with variational approaches on small-to-medium datasets

**Architecture:**
1. Encode input via angle embedding into N qubits
2. Apply L layers of random (fixed) rotations + entangling CNOTs
3. Measure expectations of all single-qubit Pauli-Z operators → N features
4. Feed quantum features into classical Ridge / Logistic Regression
""")
code("""\
N_RESERVOIR_LAYERS = 15
n_qubits = N_QUBITS

# Fixed random parameters (not trained)
np.random.seed(SEED)
reservoir_params = np.random.uniform(0, 2*np.pi, (N_RESERVOIR_LAYERS, n_qubits, 3))

dev_qrc = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev_qrc, interface='numpy')
def reservoir_circuit(x, params):
    # Angle encoding
    for i in range(n_qubits):
        qml.RX(x[i % len(x)], wires=i)

    # Fixed random reservoir layers
    for layer in range(params.shape[0]):
        for i in range(n_qubits):
            qml.RX(params[layer, i, 0], wires=i)
            qml.RY(params[layer, i, 1], wires=i)
            qml.RZ(params[layer, i, 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def extract_reservoir_features(X_data, params, batch_label=''):
    features = []
    n = len(X_data)
    t0 = time.time()
    for idx, x in enumerate(X_data):
        x_padded = np.zeros(n_qubits)
        x_padded[:len(x)] = x[:n_qubits]
        feat = reservoir_circuit(x_padded, params)
        features.append(feat)
        if (idx+1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  {batch_label} {idx+1}/{n}  ({elapsed:.1f}s)', end='\\r')
    print(f'  {batch_label} {n}/{n} done in {time.time()-t0:.1f}s')
    return np.array(features)

print(f'QRC: {n_qubits} qubits, {N_RESERVOIR_LAYERS} layers, {reservoir_params.size} fixed params')
print(f'Extracting reservoir features...')
""")

code("""\
# ── Task 1A: QRC Classification ──────────────────────────────────────────────
# Use a subset for speed; increase on AWS
N_TRAIN_1A = min(500, len(X1_tr_q)) if not CLOUD_MODE else len(X1_tr_q)
N_TEST_1A  = min(200, len(X1_te_q)) if not CLOUD_MODE else len(X1_te_q)

X1_tr_sub = X1_tr_q[:N_TRAIN_1A]
y1_tr_sub = y1_tr[:N_TRAIN_1A]
X1_te_sub = X1_te_q[:N_TEST_1A]
y1_te_sub = y1_te[:N_TEST_1A]

t0 = time.time()
qrc_feat_tr_1 = extract_reservoir_features(X1_tr_sub, reservoir_params, 'Train-1A')
qrc_feat_te_1 = extract_reservoir_features(X1_te_sub, reservoir_params, 'Test-1A')
qrc_time_1a = time.time() - t0

# Classical readout
clf_qrc = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
clf_qrc.fit(qrc_feat_tr_1, y1_tr_sub)

y1_pred_qrc = clf_qrc.predict(qrc_feat_te_1)
y1_prob_qrc = clf_qrc.predict_proba(qrc_feat_te_1)[:, 1]

f1_qrc = f1_score(y1_te_sub, y1_pred_qrc)
auc_qrc = roc_auc_score(y1_te_sub, y1_prob_qrc)

print(f'\\n=== QRC — Task 1A (Classification) ===')
print(f'  Train: {N_TRAIN_1A}, Test: {N_TEST_1A}')
print(f'  F1 Score:  {f1_qrc:.4f}')
print(f'  AUC-ROC:   {auc_qrc:.4f}')
print(f'  Time:      {qrc_time_1a:.1f}s')
print(classification_report(y1_te_sub, y1_pred_qrc, target_names=['No Fire', 'Fire']))
""")

code("""\
# ── Task 2: QRC Regression ───────────────────────────────────────────────────
N_TRAIN_2 = min(400, len(X2_tr_q)) if not CLOUD_MODE else len(X2_tr_q)
N_TEST_2  = min(100, len(X2_te_q)) if not CLOUD_MODE else len(X2_te_q)

X2_tr_sub = X2_tr_q[:N_TRAIN_2]
y2_tr_sub = y2_tr[:N_TRAIN_2]
X2_te_sub = X2_te_q[:N_TEST_2]
y2_te_sub = y2_te[:N_TEST_2]

t0 = time.time()
qrc_feat_tr_2 = extract_reservoir_features(X2_tr_sub, reservoir_params, 'Train-T2')
qrc_feat_te_2 = extract_reservoir_features(X2_te_sub, reservoir_params, 'Test-T2')
qrc_time_t2 = time.time() - t0

reg_qrc = Ridge(alpha=1.0)
reg_qrc.fit(qrc_feat_tr_2, y2_tr_sub)

y2_pred_qrc = reg_qrc.predict(qrc_feat_te_2)
r2_qrc = r2_score(y2_te_sub, y2_pred_qrc)
rmse_qrc = np.sqrt(mean_squared_error(y2_te_sub, y2_pred_qrc))

print(f'\\n=== QRC — Task 2 (Regression, log-scale) ===')
print(f'  Train: {N_TRAIN_2}, Test: {N_TEST_2}')
print(f'  R2 Score:  {r2_qrc:.4f}')
print(f'  RMSE:      {rmse_qrc:.4f}')
print(f'  Time:      {qrc_time_t2:.1f}s')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: Trainable Quantum Kernel
# ═══════════════════════════════════════════════════════════════════════════════
md("""\
---
# Section B — Trainable Quantum Kernel

**Upgrade over Notebook 01's fixed kernel:**
- Feature map parameters are **optimized** to maximize kernel-target alignment
- Uses PennyLane's `qml.kernels.kernel_matrix` with Adam optimizer
- Targets beating classical SVM's F1 = 0.7089

**Kernel-target alignment** measures how well the quantum kernel matrix
matches the ideal kernel (outer product of labels). Higher alignment →
better classification.
""")
code("""\
n_kernel_qubits = min(N_QUBITS, 6)
dev_kernel = qml.device('default.qubit', wires=n_kernel_qubits)

# Trainable feature map
@qml.qnode(dev_kernel, interface='autograd')
def kernel_circuit(x1, x2, params):
    # Encode x1
    for i in range(n_kernel_qubits):
        qml.RY(x1[i % len(x1)] * params[i], wires=i)
        qml.RZ(x1[i % len(x1)] * params[n_kernel_qubits + i], wires=i)
    for i in range(n_kernel_qubits - 1):
        qml.CNOT(wires=[i, i+1])

    # Encode x1†  (adjoint)
    for i in reversed(range(n_kernel_qubits - 1)):
        qml.CNOT(wires=[i, i+1])
    for i in reversed(range(n_kernel_qubits)):
        qml.RZ(-x2[i % len(x2)] * params[n_kernel_qubits + i], wires=i)
        qml.RY(-x2[i % len(x2)] * params[i], wires=i)

    return qml.probs(wires=range(n_kernel_qubits))

def quantum_kernel(x1, x2, params):
    probs = kernel_circuit(x1, x2, params)
    return probs[0]  # probability of |00...0>

def kernel_matrix(X, params):
    n = len(X)
    K_rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(pnp.array(1.0))
            elif j > i:
                row.append(quantum_kernel(X[i], X[j], params))
            else:
                row.append(pnp.array(0.0))  # placeholder
        K_rows.append(pnp.stack(row))
    K = pnp.stack(K_rows)
    K = K + K.T - pnp.diag(pnp.diag(K))
    return K

def kernel_target_alignment(K, y):
    y = pnp.array(y, dtype=float)
    y_outer = pnp.outer(y, y)
    kta = pnp.sum(K * y_outer) / (pnp.sqrt(pnp.sum(K * K)) * pnp.sqrt(pnp.sum(y_outer * y_outer)))
    return kta

print(f'Trainable Kernel: {n_kernel_qubits} qubits, {2*n_kernel_qubits} trainable params')
""")

code("""\
# Use small subset for kernel training (kernel matrix is O(n^2))
N_KERNEL = min(60, len(X1_tr_q)) if not CLOUD_MODE else min(300, len(X1_tr_q))
N_KERNEL_TE = min(30, len(X1_te_q))

X_k_tr = pnp.array(X1_tr_q[:N_KERNEL, :n_kernel_qubits], requires_grad=False)
y_k_tr = y1_tr[:N_KERNEL].copy()
y_k_tr_signed = 2 * y_k_tr - 1  # {0,1} -> {-1,1}
X_k_te = pnp.array(X1_te_q[:N_KERNEL_TE, :n_kernel_qubits], requires_grad=False)
y_k_te = y1_te[:N_KERNEL_TE]

# Initialize kernel parameters
params_k = pnp.array(np.random.uniform(0.5, 1.5, 2 * n_kernel_qubits), requires_grad=True)

opt_k = qml.AdamOptimizer(stepsize=0.1)
n_kernel_epochs = 8 if not CLOUD_MODE else 30

print(f'Training quantum kernel ({N_KERNEL} samples, {n_kernel_epochs} epochs)...')
t0 = time.time()
for epoch in range(n_kernel_epochs):
    def cost_fn(p):
        K = kernel_matrix(X_k_tr, p)
        return -kernel_target_alignment(K, y_k_tr_signed)

    params_k, cost = opt_k.step_and_cost(cost_fn, params_k)
    if (epoch + 1) % 5 == 0:
        print(f'  Epoch {epoch+1:3d}  |  alignment = {-cost:.4f}  |  {time.time()-t0:.1f}s')

kernel_train_time = time.time() - t0
print(f'Kernel training done in {kernel_train_time:.1f}s')
print(f'Final kernel-target alignment: {-cost:.4f}')
""")

code("""\
# Compute kernel matrices and classify with SVM
print('Computing kernel matrices for SVM...')
K_train = np.zeros((N_KERNEL, N_KERNEL))
for i in range(N_KERNEL):
    for j in range(i, N_KERNEL):
        k = float(quantum_kernel(X_k_tr[i], X_k_tr[j], params_k))
        K_train[i, j] = k
        K_train[j, i] = k

K_test = np.zeros((N_KERNEL_TE, N_KERNEL))
for i in range(N_KERNEL_TE):
    for j in range(N_KERNEL):
        K_test[i, j] = float(quantum_kernel(X_k_te[i], X_k_tr[j], params_k))

svm_qk = SVC(kernel='precomputed', class_weight='balanced', probability=True)
svm_qk.fit(K_train, y_k_tr)

y_pred_qk = svm_qk.predict(K_test)
y_prob_qk = svm_qk.predict_proba(K_test)[:, 1]
f1_qk = f1_score(y_k_te, y_pred_qk)
auc_qk = roc_auc_score(y_k_te, y_prob_qk)

print(f'\\n=== Trainable Quantum Kernel — Task 1A ===')
print(f'  Train: {N_KERNEL}, Test: {N_KERNEL_TE}')
print(f'  F1 Score:  {f1_qk:.4f}')
print(f'  AUC-ROC:   {auc_qk:.4f}')
print(f'  Time:      {kernel_train_time:.1f}s')
print(classification_report(y_k_te, y_pred_qk, target_names=['No Fire', 'Fire']))
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: QLSTM
# ═══════════════════════════════════════════════════════════════════════════════
md("""\
---
# Section C — Quantum LSTM (QLSTM) for Time Series

**Why QLSTM?**
Task 2 explicitly asks for "time series analysis using QML." This replaces the standard
regression approach with a proper temporal model.

**Architecture:**
- Classical LSTM cell where the internal gates are replaced by variational quantum circuits
- PennyLane `qml.qnn.TorchLayer` wraps quantum circuits into PyTorch modules
- Input sequence: yearly features (2018, 2019, 2020) → predict 2021 premium
- Adam optimizer via PyTorch autograd
""")
code("""\
n_qlstm_qubits = 4
dev_qlstm = qml.device('default.qubit', wires=n_qlstm_qubits)

@qml.qnode(dev_qlstm, interface='torch')
def qlstm_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qlstm_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qlstm_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qlstm_qubits)]

weight_shapes = {'weights': (3, n_qlstm_qubits, 3)}

class QLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits

        self.proj_in = nn.Linear(input_size + hidden_size, n_qubits)
        self.q_forget = qml.qnn.TorchLayer(qlstm_circuit, weight_shapes)
        self.q_input  = qml.qnn.TorchLayer(qlstm_circuit, weight_shapes)
        self.q_cell   = qml.qnn.TorchLayer(qlstm_circuit, weight_shapes)
        self.q_output = qml.qnn.TorchLayer(qlstm_circuit, weight_shapes)
        self.proj_out = nn.Linear(n_qubits, hidden_size)

    def forward(self, x, states):
        h, c = states
        combined = torch.cat([x, h], dim=-1)
        q_in = torch.sigmoid(self.proj_in(combined))

        f = torch.sigmoid(self.proj_out(self.q_forget(q_in)))
        i = torch.sigmoid(self.proj_out(self.q_input(q_in)))
        g = torch.tanh(self.proj_out(self.q_cell(q_in)))
        o = torch.sigmoid(self.proj_out(self.q_output(q_in)))

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

class QLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = QLSTMCell(input_size, hidden_size, n_qubits)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_seq):
        batch_size = x_seq.size(0)
        h = torch.zeros(batch_size, self.hidden_size)
        c = torch.zeros(batch_size, self.hidden_size)
        for t in range(x_seq.size(1)):
            h, c = self.cell(x_seq[:, t, :], (h, c))
        return self.fc(h)

print(f'QLSTM: {n_qlstm_qubits} qubits, 4 quantum gates per cell, StronglyEntanglingLayers')
""")

code("""\
# ── Build time series sequences from insurance data ──────────────────────────
# Group by ZIP, create sequences of yearly features -> next year premium
ins_ts = ins.copy()
ins_ts = ins_ts.dropna(subset=T2_FEATURES + ['Year'])
ins_ts = ins_ts.sort_values(['ZIP_Code', 'Year'])

feature_cols_ts = T2_FEATURES
sc_ts = StandardScaler()
ins_ts[feature_cols_ts] = sc_ts.fit_transform(ins_ts[feature_cols_ts])

# Build sequences: [year_t, year_t+1, year_t+2] -> premium at year_t+3
sequences, targets = [], []
for zc, grp in ins_ts.groupby('ZIP_Code'):
    grp = grp.sort_values('Year')
    if len(grp) < 3:
        continue
    years_list = grp['Year'].values
    feats = grp[feature_cols_ts].values
    prems = grp[col_prem].values
    for i in range(len(grp) - 2):
        seq = feats[i:i+2]  # 2-step sequence
        tgt = np.log1p(np.abs(prems[i+2])) * np.sign(prems[i+2])
        sequences.append(seq)
        targets.append(tgt)

sequences = np.array(sequences, dtype=np.float32)
targets = np.array(targets, dtype=np.float32)

# Train/test split
n_ts = len(sequences)
split_idx = int(0.8 * n_ts)
X_ts_tr, X_ts_te = sequences[:split_idx], sequences[split_idx:]
y_ts_tr, y_ts_te = targets[:split_idx], targets[split_idx:]

# PCA reduce features to n_qlstm_qubits
X_ts_flat_tr = X_ts_tr.reshape(-1, X_ts_tr.shape[-1])
pca_ts = PCA(n_components=n_qlstm_qubits)
pca_ts.fit(X_ts_flat_tr)

X_ts_tr_q = np.array([pca_ts.transform(s) for s in X_ts_tr], dtype=np.float32)
X_ts_te_q = np.array([pca_ts.transform(s) for s in X_ts_te], dtype=np.float32)

# Scale to [0, pi]
mm_ts = MinMaxScaler(feature_range=(0, np.pi))
shape_tr = X_ts_tr_q.shape
shape_te = X_ts_te_q.shape
X_ts_tr_q = mm_ts.fit_transform(X_ts_tr_q.reshape(-1, n_qlstm_qubits)).reshape(shape_tr)
X_ts_te_q = mm_ts.transform(X_ts_te_q.reshape(-1, n_qlstm_qubits)).reshape(shape_te)

print(f'Time series sequences: {n_ts} total')
print(f'  Train: {len(X_ts_tr_q)} sequences of shape {X_ts_tr_q.shape[1:]}')
print(f'  Test:  {len(X_ts_te_q)} sequences of shape {X_ts_te_q.shape[1:]}')
""")

code("""\
# ── Train QLSTM ─────────────────────────────────────────────────────────────
HIDDEN = 8
QLSTM_EPOCHS = 30 if not CLOUD_MODE else 80
QLSTM_LR = 0.01
QLSTM_BATCH = 32
N_TS_TRAIN = min(200, len(X_ts_tr_q)) if not CLOUD_MODE else len(X_ts_tr_q)

model_qlstm = QLSTMModel(n_qlstm_qubits, HIDDEN, n_qlstm_qubits)
optimizer_qlstm = torch.optim.Adam(model_qlstm.parameters(), lr=QLSTM_LR)
criterion = nn.MSELoss()

X_t = torch.tensor(X_ts_tr_q[:N_TS_TRAIN])
y_t = torch.tensor(y_ts_tr[:N_TS_TRAIN]).unsqueeze(-1)
X_te_t = torch.tensor(X_ts_te_q)
y_te_t = torch.tensor(y_ts_te).unsqueeze(-1)

dataset = TensorDataset(X_t, y_t)
loader = DataLoader(dataset, batch_size=QLSTM_BATCH, shuffle=True)

print(f'Training QLSTM ({N_TS_TRAIN} samples, {QLSTM_EPOCHS} epochs)...')
t0 = time.time()
losses = []
for epoch in range(QLSTM_EPOCHS):
    epoch_loss = 0
    for xb, yb in loader:
        optimizer_qlstm.zero_grad()
        pred = model_qlstm(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer_qlstm.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(loader))
    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1:3d}  |  loss = {losses[-1]:.4f}  |  {time.time()-t0:.1f}s')

qlstm_time = time.time() - t0

# Evaluate
model_qlstm.eval()
with torch.no_grad():
    y_pred_qlstm = model_qlstm(X_te_t).numpy().flatten()
y_actual = y_ts_te

r2_qlstm = r2_score(y_actual, y_pred_qlstm)
rmse_qlstm = np.sqrt(mean_squared_error(y_actual, y_pred_qlstm))

print(f'\\n=== QLSTM — Task 2 (Time Series, log-scale) ===')
print(f'  R2 Score:  {r2_qlstm:.4f}')
print(f'  RMSE:      {rmse_qlstm:.4f}')
print(f'  Time:      {qlstm_time:.1f}s')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D: Transfer Learning
# ═══════════════════════════════════════════════════════════════════════════════
md("""\
---
# Section D — Quantum Transfer Learning

**Approach:** Pre-trained classical MLP extracts features → quantum circuit head for final prediction.
This leverages classical networks' ability to learn basic representations while the quantum
circuit adds expressivity for the final decision boundary.
""")
code("""\
n_tl_qubits = 4
dev_tl = qml.device('default.qubit', wires=n_tl_qubits)

@qml.qnode(dev_tl, interface='torch')
def transfer_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_tl_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_tl_qubits))
    return qml.expval(qml.PauliZ(0))

tl_weight_shapes = {'weights': (2, n_tl_qubits, 3)}

class TransferModel(nn.Module):
    def __init__(self, input_dim, hidden=32, n_qubits=4):
        super().__init__()
        self.classical = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_qubits),
            nn.Sigmoid(),  # scale to [0, 1]
        )
        self.scale = nn.Parameter(torch.tensor(np.pi))
        self.quantum = qml.qnn.TorchLayer(transfer_circuit, tl_weight_shapes)
        self.fc_out = nn.Linear(1, 1)

    def forward(self, x):
        h = self.classical(x) * self.scale  # scale to [0, pi]
        q_out = self.quantum(h).unsqueeze(-1)
        return torch.sigmoid(self.fc_out(q_out)).squeeze(-1)

# Task 1A transfer learning
X_tl_tr = torch.tensor(X1_tr_s[:500].astype(np.float32) if not CLOUD_MODE
                        else X1_tr_s.astype(np.float32))
y_tl_tr = torch.tensor(y1_tr[:500].astype(np.float32) if not CLOUD_MODE
                        else y1_tr.astype(np.float32))
X_tl_te = torch.tensor(X1_te_s.astype(np.float32))
y_tl_te_np = y1_te

model_tl = TransferModel(X_tl_tr.shape[1], hidden=32, n_qubits=n_tl_qubits)
opt_tl = torch.optim.Adam(model_tl.parameters(), lr=0.005)
criterion_tl = nn.BCELoss()

TL_EPOCHS = 30 if not CLOUD_MODE else 60
print(f'Training Transfer Model ({len(X_tl_tr)} samples, {TL_EPOCHS} epochs)...')
t0 = time.time()
for epoch in range(TL_EPOCHS):
    model_tl.train()
    opt_tl.zero_grad()
    pred = model_tl(X_tl_tr)
    loss = criterion_tl(pred, y_tl_tr)
    loss.backward()
    opt_tl.step()
    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1:3d}  |  loss = {loss.item():.4f}  |  {time.time()-t0:.1f}s')

tl_time = time.time() - t0

model_tl.eval()
with torch.no_grad():
    y_prob_tl = model_tl(X_tl_te).numpy()
y_pred_tl = (y_prob_tl > 0.5).astype(int)

f1_tl = f1_score(y_tl_te_np, y_pred_tl)
auc_tl = roc_auc_score(y_tl_te_np, y_prob_tl)

print(f'\\n=== Transfer Learning — Task 1A ===')
print(f'  F1 Score:  {f1_tl:.4f}')
print(f'  AUC-ROC:   {auc_tl:.4f}')
print(f'  Time:      {tl_time:.1f}s')
print(classification_report(y_tl_te_np, y_pred_tl, target_names=['No Fire', 'Fire']))
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E: Real Hardware
# ═══════════════════════════════════════════════════════════════════════════════
md("""\
---
# Section E — Real Quantum Hardware (AWS Braket)

**Instructions for your 48-hour AWS window:**
1. Set `CLOUD_MODE = True` and `RUN_HARDWARE = True` at the top
2. Configure your S3 bucket and device ARN below
3. Run the cell — it will submit the best model to real hardware
4. Even 50 samples on real hardware is a massive differentiator

**Available devices:**
- `arn:aws:braket:::device/qpu/ionq/Aria-1` — IonQ Aria (trapped ion, 25 qubits)
- `arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3` — Rigetti Ankaa (superconducting)
- `arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet` — IQM Garnet
""")
code("""\
RUN_HARDWARE = False  # Set True only on AWS Braket with credits

if RUN_HARDWARE and CLOUD_MODE:
    S3_BUCKET = 'your-braket-bucket'
    S3_PREFIX = 'qsc-challenge'
    DEVICE_ARN = 'arn:aws:braket:::device/qpu/ionq/Aria-1'

    dev_hw = qml.device(
        'braket.aws.qubit',
        device_arn=DEVICE_ARN,
        wires=n_kernel_qubits,
        s3_destination_folder=(S3_BUCKET, S3_PREFIX),
        shots=1000,
    )

    @qml.qnode(dev_hw, interface='numpy')
    def hw_kernel_circuit(x1, x2, params):
        for i in range(n_kernel_qubits):
            qml.RY(x1[i % len(x1)] * params[i], wires=i)
        for i in range(n_kernel_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        for i in reversed(range(n_kernel_qubits - 1)):
            qml.CNOT(wires=[i, i+1])
        for i in reversed(range(n_kernel_qubits)):
            qml.RY(-x2[i % len(x2)] * params[i], wires=i)
        return qml.probs(wires=range(n_kernel_qubits))

    N_HW = 50
    print(f'Running {N_HW} samples on real hardware: {DEVICE_ARN}')
    print('This may take several minutes per sample...')

    hw_results = []
    t0 = time.time()
    for i in range(N_HW):
        probs = hw_kernel_circuit(X_k_te[i], X_k_tr[0], params_k)
        hw_results.append(float(probs[0]))
        if (i+1) % 10 == 0:
            print(f'  {i+1}/{N_HW} done ({time.time()-t0:.1f}s)')

    print(f'Hardware execution: {time.time()-t0:.1f}s for {N_HW} kernel evaluations')
    print(f'Mean kernel value: {np.mean(hw_results):.4f}')
    print(f'Std kernel value:  {np.std(hw_results):.4f}')
else:
    print('Hardware run skipped (set RUN_HARDWARE=True and CLOUD_MODE=True on AWS)')
    print('Simulator results from sections A-D are sufficient for submission')
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F: Results Comparison
# ═══════════════════════════════════════════════════════════════════════════════
md("""\
---
# Section F — Results Comparison & Resource Requirements
""")
code("""\
# ── Classical baselines for fair comparison ──────────────────────────────────
# Task 1A classical (on same subsets)
clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
clf_lr.fit(X1_tr_s[:N_TRAIN_1A], y1_tr[:N_TRAIN_1A])
y_lr_1a = clf_lr.predict(X1_te_s[:N_TEST_1A])
f1_lr = f1_score(y1_te[:N_TEST_1A], y_lr_1a)

clf_rf = RandomForestClassifier(200, class_weight='balanced', random_state=SEED)
clf_rf.fit(X1_tr_s[:N_TRAIN_1A], y1_tr[:N_TRAIN_1A])
y_rf_1a = clf_rf.predict(X1_te_s[:N_TEST_1A])
f1_rf = f1_score(y1_te[:N_TEST_1A], y_rf_1a)

clf_xgb = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False,
                              eval_metric='logloss', random_state=SEED)
clf_xgb.fit(X1_tr_s[:N_TRAIN_1A], y1_tr[:N_TRAIN_1A])
y_xgb_1a = clf_xgb.predict(X1_te_s[:N_TEST_1A])
f1_xgb = f1_score(y1_te[:N_TEST_1A], y_xgb_1a)

# Task 2 classical
reg_lr2 = Ridge(alpha=1.0)
reg_lr2.fit(X2_tr_s[:N_TRAIN_2], y2_tr[:N_TRAIN_2])
y_lr2 = reg_lr2.predict(X2_te_s[:N_TEST_2])
r2_lr2 = r2_score(y2_te[:N_TEST_2], y_lr2)

print('=== COMPREHENSIVE RESULTS COMPARISON ===\\n')

print('TASK 1A — Wildfire Classification (F1 Score)')
print(f'  {"Model":<30s} {"Type":<12s} {"F1":>8s} {"Qubits":>8s}')
print(f'  {"-"*60}')
print(f'  {"Logistic Regression":<30s} {"Classical":<12s} {f1_lr:>8.4f} {"—":>8s}')
print(f'  {"Random Forest":<30s} {"Classical":<12s} {f1_rf:>8.4f} {"—":>8s}')
print(f'  {"XGBoost":<30s} {"Classical":<12s} {f1_xgb:>8.4f} {"—":>8s}')
print(f'  {"QRC + LogReg":<30s} {"Quantum":<12s} {f1_qrc:>8.4f} {N_QUBITS:>8d}')
print(f'  {"Trainable Quantum Kernel+SVM":<30s} {"Quantum":<12s} {f1_qk:>8.4f} {n_kernel_qubits:>8d}')
print(f'  {"Quantum Transfer Learning":<30s} {"Hybrid":<12s} {f1_tl:>8.4f} {n_tl_qubits:>8d}')

print(f'\\nTASK 2 — Premium Regression (R2 Score, log-scale)')
print(f'  {"Model":<30s} {"Type":<12s} {"R2":>8s} {"Qubits":>8s}')
print(f'  {"-"*60}')
print(f'  {"Ridge Regression":<30s} {"Classical":<12s} {r2_lr2:>8.4f} {"—":>8s}')
print(f'  {"QRC + Ridge":<30s} {"Quantum":<12s} {r2_qrc:>8.4f} {N_QUBITS:>8d}')
print(f'  {"QLSTM (time series)":<30s} {"Quantum":<12s} {r2_qlstm:>8.4f} {n_qlstm_qubits:>8d}')
""")

code("""\
# ── Visualization ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Task 1A bar chart
models_1a = ['LogReg', 'RF', 'XGBoost', 'QRC', 'Trainable\\nKernel', 'Transfer\\nLearning']
f1s = [f1_lr, f1_rf, f1_xgb, f1_qrc, f1_qk, f1_tl]
colors_1a = ['#78909C', '#78909C', '#78909C', '#1976D2', '#1976D2', '#7B1FA2']
bars = axes[0].bar(models_1a, f1s, color=colors_1a, edgecolor='white')
for bar, v in zip(bars, f1s):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[0].set(title='Task 1A: Wildfire Classification (F1 Score)', ylabel='F1 Score', ylim=[0, 1])
axes[0].axhline(y=max(f1_lr, f1_rf, f1_xgb), color='gray', ls='--', alpha=0.5, label='Best Classical')
axes[0].legend(fontsize=8)

# Task 2 bar chart
models_2 = ['Ridge', 'QRC', 'QLSTM']
r2s = [r2_lr2, r2_qrc, r2_qlstm]
colors_2 = ['#78909C', '#1976D2', '#1976D2']
bars = axes[1].bar(models_2, r2s, color=colors_2, edgecolor='white')
for bar, v in zip(bars, r2s):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 max(bar.get_height(), 0) + 0.02,
                 f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[1].set(title='Task 2: Premium Prediction (R2 Score, log-scale)', ylabel='R2 Score')
axes[1].axhline(y=r2_lr2, color='gray', ls='--', alpha=0.5, label='Classical Ridge')
axes[1].legend(fontsize=8)

plt.suptitle('Quantum vs Classical — Advanced Models Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('../results/advanced_quantum_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
""")

code("""\
# ── Resource Requirements Table ──────────────────────────────────────────────
resources = pd.DataFrame([
    {'Model': 'QRC (Task 1A)', 'Qubits': N_QUBITS, 'Circuit Depth': N_RESERVOIR_LAYERS*4,
     'Trainable Params': 0, 'Fixed Params': reservoir_params.size,
     'Training Time (s)': round(qrc_time_1a, 1)},
    {'Model': 'QRC (Task 2)', 'Qubits': N_QUBITS, 'Circuit Depth': N_RESERVOIR_LAYERS*4,
     'Trainable Params': 0, 'Fixed Params': reservoir_params.size,
     'Training Time (s)': round(qrc_time_t2, 1)},
    {'Model': 'Trainable Kernel', 'Qubits': n_kernel_qubits,
     'Circuit Depth': n_kernel_qubits*2,
     'Trainable Params': 2*n_kernel_qubits, 'Fixed Params': 0,
     'Training Time (s)': round(kernel_train_time, 1)},
    {'Model': 'QLSTM', 'Qubits': n_qlstm_qubits, 'Circuit Depth': 12,
     'Trainable Params': '4x(3x4x3)=144', 'Fixed Params': 0,
     'Training Time (s)': round(qlstm_time, 1)},
    {'Model': 'Transfer Learning', 'Qubits': n_tl_qubits, 'Circuit Depth': 8,
     'Trainable Params': '24 quantum + MLP', 'Fixed Params': 0,
     'Training Time (s)': round(tl_time, 1)},
])
print('\\n=== QUANTUM RESOURCE REQUIREMENTS ===')
print(resources.to_string(index=False))

# Save results
results = {
    'task_1a': {
        'logistic_regression': {'f1': round(f1_lr, 4), 'type': 'classical'},
        'random_forest': {'f1': round(f1_rf, 4), 'type': 'classical'},
        'xgboost': {'f1': round(f1_xgb, 4), 'type': 'classical'},
        'qrc': {'f1': round(f1_qrc, 4), 'qubits': N_QUBITS, 'type': 'quantum'},
        'trainable_kernel': {'f1': round(f1_qk, 4), 'qubits': n_kernel_qubits, 'type': 'quantum'},
        'transfer_learning': {'f1': round(f1_tl, 4), 'qubits': n_tl_qubits, 'type': 'hybrid'},
    },
    'task_2': {
        'ridge': {'r2': round(r2_lr2, 4), 'type': 'classical'},
        'qrc': {'r2': round(r2_qrc, 4), 'qubits': N_QUBITS, 'type': 'quantum'},
        'qlstm': {'r2': round(r2_qlstm, 4), 'qubits': n_qlstm_qubits, 'type': 'quantum'},
    }
}
with open('../results/advanced_quantum_results.json', 'w') as f:
    _json.dump(results, f, indent=2)
resources.to_csv('../results/quantum_resource_requirements.csv', index=False)
print('\\nResults saved to results/advanced_quantum_results.json')
print('Resources saved to results/quantum_resource_requirements.csv')
""")

md("""\
---
# Section G — Time-Series Hardening: Fair Classical Baselines

The QLSTM result above (R2 = see above) is tested on the **same** time-series sequences
(2-step ZIP-code yearly features → next-year premium). For honest comparison, we train
classical models on the **exact same** data splits: same sequences, same train/test split,
same PCA-reduced features. This ensures the only variable is the model architecture.
""")

code("""\
# ── Classical LSTM baseline on the same time-series data ──────────────────────
class ClassicalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))

model_clstm = ClassicalLSTM(n_qlstm_qubits, HIDDEN)
opt_clstm = torch.optim.Adam(model_clstm.parameters(), lr=QLSTM_LR)
dataset_cl = TensorDataset(X_t, y_t)
loader_cl = DataLoader(dataset_cl, batch_size=QLSTM_BATCH, shuffle=True)

print(f'Training Classical LSTM ({N_TS_TRAIN} samples, {QLSTM_EPOCHS} epochs)...')
t0 = time.time()
for epoch in range(QLSTM_EPOCHS):
    epoch_loss = 0
    for xb, yb in loader_cl:
        opt_clstm.zero_grad()
        pred = model_clstm(xb)
        loss = criterion(pred, yb)
        loss.backward()
        opt_clstm.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1:3d}  |  loss = {epoch_loss/len(loader_cl):.4f}  |  {time.time()-t0:.1f}s')
clstm_time = time.time() - t0

model_clstm.eval()
with torch.no_grad():
    y_pred_clstm = model_clstm(X_te_t).numpy().flatten()
r2_clstm = r2_score(y_actual, y_pred_clstm)
rmse_clstm = np.sqrt(mean_squared_error(y_actual, y_pred_clstm))

print(f'\\n=== Classical LSTM — Task 2 (same data as QLSTM) ===')
print(f'  R2 Score:  {r2_clstm:.4f}')
print(f'  RMSE:      {rmse_clstm:.4f}')
print(f'  Time:      {clstm_time:.1f}s')

# ── Temporal XGBoost baseline ────────────────────────────────────────────────
X_ts_tr_flat = X_ts_tr_q[:N_TS_TRAIN].reshape(N_TS_TRAIN, -1)
X_ts_te_flat = X_ts_te_q.reshape(len(X_ts_te_q), -1)

xgb_ts = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=SEED)
xgb_ts.fit(X_ts_tr_flat, y_ts_tr[:N_TS_TRAIN])
y_pred_xgb_ts = xgb_ts.predict(X_ts_te_flat)

r2_xgb_ts = r2_score(y_actual, y_pred_xgb_ts)
rmse_xgb_ts = np.sqrt(mean_squared_error(y_actual, y_pred_xgb_ts))
print(f'\\n=== Temporal XGBoost — Task 2 (same data as QLSTM) ===')
print(f'  R2 Score:  {r2_xgb_ts:.4f}')
print(f'  RMSE:      {rmse_xgb_ts:.4f}')

# ── Fair comparison table ────────────────────────────────────────────────────
print(f'\\n=== FAIR TIME-SERIES COMPARISON (identical data splits) ===')
print(f'  {"Model":<25s} {"Type":<12s} {"R2":>8s} {"RMSE":>8s} {"Qubits":>8s}')
print(f'  {"-"*65}')
print(f'  {"Classical LSTM":<25s} {"Classical":<12s} {r2_clstm:>8.4f} {rmse_clstm:>8.4f} {"—":>8s}')
print(f'  {"Temporal XGBoost":<25s} {"Classical":<12s} {r2_xgb_ts:>8.4f} {rmse_xgb_ts:>8.4f} {"—":>8s}')
print(f'  {"QLSTM (4 qubits)":<25s} {"Quantum":<12s} {r2_qlstm:>8.4f} {rmse_qlstm:>8.4f} {n_qlstm_qubits:>8d}')
""")

md("""\
---
# Section H — Optuna Hyperparameter Tuning

We use Optuna to automatically find the best hyperparameters for the classical models that
are compared against quantum models. This ensures the classical baselines are as strong as
possible — making any quantum advantage claim more credible.

**Models tuned (50 trials, 3-fold CV each):**
1. XGBoost (Task 1A classification) — objective: F1
2. XGBoost (Task 2 regression) — objective: R2
3. QRC + LogisticRegression head (Task 1A) — objective: F1
4. QRC + Ridge head (Task 2) — objective: R2
""")

code("""\
from sklearn.model_selection import cross_val_score

# ── 1. Optuna: XGBoost Task 1A ──────────────────────────────────────────────
def xgb_1a_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    clf = xgb.XGBClassifier(**params, use_label_encoder=False,
                             eval_metric='logloss', random_state=SEED)
    scores = cross_val_score(clf, X1_tr_s[:N_TRAIN_1A], y1_tr[:N_TRAIN_1A],
                             cv=3, scoring='f1')
    return scores.mean()

print('Optuna: tuning XGBoost (Task 1A, 50 trials)...')
t0 = time.time()
study_xgb_1a = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb_1a.optimize(xgb_1a_objective, n_trials=50, show_progress_bar=False)
print(f'  Best F1 (CV): {study_xgb_1a.best_value:.4f}  |  {time.time()-t0:.1f}s')
print(f'  Best params: {study_xgb_1a.best_params}')

clf_xgb_tuned = xgb.XGBClassifier(**study_xgb_1a.best_params, use_label_encoder=False,
                                    eval_metric='logloss', random_state=SEED)
clf_xgb_tuned.fit(X1_tr_s[:N_TRAIN_1A], y1_tr[:N_TRAIN_1A])
y_xgb_tuned_1a = clf_xgb_tuned.predict(X1_te_s[:N_TEST_1A])
f1_xgb_tuned = f1_score(y1_te[:N_TEST_1A], y_xgb_tuned_1a)
print(f'  Tuned XGBoost F1 (test): {f1_xgb_tuned:.4f}  (was {f1_xgb:.4f})')

# ── 2. Optuna: XGBoost Task 2 ───────────────────────────────────────────────
def xgb_t2_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    reg = xgb.XGBRegressor(**params, random_state=SEED)
    scores = cross_val_score(reg, X2_tr_s[:N_TRAIN_2], y2_tr[:N_TRAIN_2],
                             cv=3, scoring='r2')
    return scores.mean()

print('\\nOptuna: tuning XGBoost (Task 2, 50 trials)...')
t0 = time.time()
study_xgb_t2 = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb_t2.optimize(xgb_t2_objective, n_trials=50, show_progress_bar=False)
print(f'  Best R2 (CV): {study_xgb_t2.best_value:.4f}  |  {time.time()-t0:.1f}s')
print(f'  Best params: {study_xgb_t2.best_params}')

reg_xgb_tuned = xgb.XGBRegressor(**study_xgb_t2.best_params, random_state=SEED)
reg_xgb_tuned.fit(X2_tr_s[:N_TRAIN_2], y2_tr[:N_TRAIN_2])
y_xgb_tuned_t2 = reg_xgb_tuned.predict(X2_te_s[:N_TEST_2])
r2_xgb_tuned = r2_score(y2_te[:N_TEST_2], y_xgb_tuned_t2)
print(f'  Tuned XGBoost R2 (test): {r2_xgb_tuned:.4f}  (was {r2_lr2:.4f} Ridge)')

# ── 3. Optuna: QRC + LogReg head (Task 1A) ──────────────────────────────────
def qrc_lr_objective(trial):
    C = trial.suggest_float('C', 0.001, 100.0, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
    clf = LogisticRegression(C=C, solver=solver, max_iter=2000,
                             class_weight='balanced', random_state=SEED)
    scores = cross_val_score(clf, qrc_feat_tr_1, y1_tr_sub,
                             cv=3, scoring='f1')
    return scores.mean()

print('\\nOptuna: tuning QRC + LogReg head (Task 1A, 50 trials)...')
t0 = time.time()
study_qrc_lr = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
study_qrc_lr.optimize(qrc_lr_objective, n_trials=50, show_progress_bar=False)
print(f'  Best F1 (CV): {study_qrc_lr.best_value:.4f}  |  {time.time()-t0:.1f}s')

clf_qrc_tuned = LogisticRegression(
    C=study_qrc_lr.best_params['C'],
    solver=study_qrc_lr.best_params['solver'],
    max_iter=2000, class_weight='balanced', random_state=SEED)
clf_qrc_tuned.fit(qrc_feat_tr_1, y1_tr_sub)
y_qrc_tuned_1a = clf_qrc_tuned.predict(qrc_feat_te_1)
f1_qrc_tuned = f1_score(y1_te_sub, y_qrc_tuned_1a)
print(f'  Tuned QRC+LogReg F1 (test): {f1_qrc_tuned:.4f}  (was {f1_qrc:.4f})')

# ── 4. Optuna: QRC + Ridge head (Task 2) ────────────────────────────────────
def qrc_ridge_objective(trial):
    alpha = trial.suggest_float('alpha', 0.001, 1000.0, log=True)
    reg = Ridge(alpha=alpha, random_state=SEED)
    scores = cross_val_score(reg, qrc_feat_tr_2, y2_tr_sub,
                             cv=3, scoring='r2')
    return scores.mean()

print('\\nOptuna: tuning QRC + Ridge head (Task 2, 50 trials)...')
t0 = time.time()
study_qrc_ridge = optuna.create_study(direction='maximize',
                                       sampler=optuna.samplers.TPESampler(seed=SEED))
study_qrc_ridge.optimize(qrc_ridge_objective, n_trials=50, show_progress_bar=False)
print(f'  Best R2 (CV): {study_qrc_ridge.best_value:.4f}  |  {time.time()-t0:.1f}s')

reg_qrc_tuned = Ridge(alpha=study_qrc_ridge.best_params['alpha'], random_state=SEED)
reg_qrc_tuned.fit(qrc_feat_tr_2, y2_tr_sub)
y_qrc_tuned_t2 = reg_qrc_tuned.predict(qrc_feat_te_2)
r2_qrc_tuned = r2_score(y2_te_sub, y_qrc_tuned_t2)
print(f'  Tuned QRC+Ridge R2 (test): {r2_qrc_tuned:.4f}  (was {r2_qrc:.4f})')

print('\\n=== OPTUNA SUMMARY ===')
print(f'  XGBoost Task 1A:   F1  {f1_xgb:.4f} -> {f1_xgb_tuned:.4f}')
print(f'  XGBoost Task 2:    R2  {r2_lr2:.4f} -> {r2_xgb_tuned:.4f}')
print(f'  QRC+LogReg Task1A: F1  {f1_qrc:.4f} -> {f1_qrc_tuned:.4f}')
print(f'  QRC+Ridge Task 2:  R2  {r2_qrc:.4f} -> {r2_qrc_tuned:.4f}')
""")

md("""\
---
# Section I — SHAP Explainability

SHAP (SHapley Additive exPlanations) reveals **which features** drive each prediction and by
how much. We apply it to both the classical XGBoost models and the QRC quantum heads, providing
interpretable explanations for quantum-enhanced predictions.
""")

code("""\
# ── SHAP: XGBoost Task 1A (Classification) ──────────────────────────────────
feature_names_1a = ['PRECIPITATION', 'MAX_TEMP', 'MIN_TEMP', 'AVG_WIND_SPEED',
                    'TEMP_RANGE', 'WIND_TEMP_RATIO', 'MONTH', 'DAY_OF_YEAR',
                    'LAGGED_PRECIP', 'LAGGED_WIND']
X_shap_1a = pd.DataFrame(X1_te_s[:N_TEST_1A],
                          columns=feature_names_1a[:X1_te_s.shape[1]])

explainer_xgb_1a = shap.TreeExplainer(clf_xgb_tuned)
shap_values_xgb_1a = explainer_xgb_1a(X_shap_1a)

fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.beeswarm(shap_values_xgb_1a, show=False, max_display=12)
plt.title('SHAP — XGBoost Task 1A: What Drives Wildfire Prediction?', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('../results/shap_xgb_task1a.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: results/shap_xgb_task1a.png')

# ── SHAP: XGBoost Task 2 (Regression) ───────────────────────────────────────
T2_FEAT_NAMES = ['Avg_Fire_Risk', 'Exposure', 'Total_Insured_Loss']
X_shap_t2 = pd.DataFrame(X2_te_s[:N_TEST_2],
                          columns=T2_FEAT_NAMES[:X2_te_s.shape[1]])

explainer_xgb_t2 = shap.TreeExplainer(reg_xgb_tuned)
shap_values_xgb_t2 = explainer_xgb_t2(X_shap_t2)

fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.beeswarm(shap_values_xgb_t2, show=False, max_display=10)
plt.title('SHAP — XGBoost Task 2: What Drives Premium Prediction?', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('../results/shap_xgb_task2.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: results/shap_xgb_task2.png')

# ── SHAP: QRC + LogReg Head (Task 1A) ───────────────────────────────────────
qrc_feat_names = [f'QRC_{i}' for i in range(qrc_feat_te_1.shape[1])]
X_shap_qrc_1a = pd.DataFrame(qrc_feat_te_1, columns=qrc_feat_names)

explainer_qrc_1a = shap.LinearExplainer(clf_qrc_tuned,
                                         pd.DataFrame(qrc_feat_tr_1, columns=qrc_feat_names))
shap_values_qrc_1a = explainer_qrc_1a(X_shap_qrc_1a)

fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.beeswarm(shap_values_qrc_1a, show=False)
plt.title('SHAP — QRC Quantum Features (Task 1A): Which Reservoir Outputs Matter?',
          fontsize=11, pad=15)
plt.tight_layout()
plt.savefig('../results/shap_qrc_task1a.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: results/shap_qrc_task1a.png')

# ── SHAP: QRC + Ridge Head (Task 2) ─────────────────────────────────────────
X_shap_qrc_t2 = pd.DataFrame(qrc_feat_te_2, columns=qrc_feat_names)

explainer_qrc_t2 = shap.LinearExplainer(reg_qrc_tuned,
                                          pd.DataFrame(qrc_feat_tr_2, columns=qrc_feat_names))
shap_values_qrc_t2 = explainer_qrc_t2(X_shap_qrc_t2)

fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.beeswarm(shap_values_qrc_t2, show=False)
plt.title('SHAP — QRC Quantum Features (Task 2): Which Reservoir Outputs Matter?',
          fontsize=11, pad=15)
plt.tight_layout()
plt.savefig('../results/shap_qrc_task2.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: results/shap_qrc_task2.png')

# ── Combined 2x2 SHAP summary ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

plt.sca(axes[0, 0])
shap.plots.beeswarm(shap_values_xgb_1a, show=False, max_display=8)
axes[0, 0].set_title('XGBoost — Task 1A (Wildfire)', fontsize=11)

plt.sca(axes[0, 1])
shap.plots.beeswarm(shap_values_xgb_t2, show=False, max_display=8)
axes[0, 1].set_title('XGBoost — Task 2 (Premium)', fontsize=11)

plt.sca(axes[1, 0])
shap.plots.beeswarm(shap_values_qrc_1a, show=False)
axes[1, 0].set_title('QRC Quantum Head — Task 1A', fontsize=11)

plt.sca(axes[1, 1])
shap.plots.beeswarm(shap_values_qrc_t2, show=False)
axes[1, 1].set_title('QRC Quantum Head — Task 2', fontsize=11)

plt.suptitle('SHAP Explainability Summary — Classical vs Quantum Features',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('../results/shap_nb05_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: results/shap_nb05_summary.png')
""")

md("""\
---
## Data Assumptions for 2026 Forecasting

**Features assumed known for future prediction:**
- ZIP-code-level fire risk scores (from state-maintained CalFire risk maps — publicly available, updated annually)
- Historical premium patterns per ZIP (from CDI regulatory filings — reported with ~1 year lag)
- Historical loss ratios and exposure counts (from CDI filings)

**Features extrapolated / estimated:**
- 2022-2025 premium trends (linear extrapolation from 2018-2021 slope)
- Future fire loss amounts (estimated from risk scores and historical loss distributions)
- Premium-per-policy growth (estimated from county-level CPI and building cost indices)

**Validation protocol:** Temporal train/test split (train on 2018-2020 data, test on 2021). For
competition claims, the 2021 test performance is the primary evidence. The 2026 predictions
are extrapolations, not validated forecasts, and should be presented with confidence intervals.
""")

md("""\
---
## Summary

| Model | Task | Metric | Qubits | Innovation |
|-------|------|--------|--------|------------|
| **QRC** | 1A, 2 | See above | 8 | No barren plateaus, trains in seconds |
| **Trainable Kernel** | 1A | See above | 6 | Optimized feature map, kernel-target alignment |
| **QLSTM** | 2 | See above | 4 | True time-series QML (challenge requirement) |
| **Transfer Learning** | 1A | See above | 4 | Classical features + quantum decision boundary |

**To run on AWS Braket:**
1. Upload this notebook to a SageMaker/Braket notebook instance
2. Set `CLOUD_MODE = True` at the top
3. For real hardware: set `RUN_HARDWARE = True` and configure S3/device ARN in Section E
4. Increase sample sizes (remove the `if not CLOUD_MODE` caps)

**Next:** Use these models' predictions in Notebook 06 for final 2026 ZIP-code risk mapping.
""")

# ═══════════════════════════════════════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4, "nbformat_minor": 4
}

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "notebooks", "05_Cloud_Quantum_Models.ipynb")
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Written: {outpath}")
