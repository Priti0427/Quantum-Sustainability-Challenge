"""
Cloud-Ready Advanced Quantum Models (GPU Accelerated - FULL LOCAL RUN)
----------------------------------------------------------------------
This script implements four quantum ML experiments with full GPU acceleration.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import json as _json

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
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ==========================================
# 0 - Setup & Cloud Configuration
# ==========================================
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 150})
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Compute Device: {device}")

N_QUBITS = 8

LOCAL_Q_DEV = 'default.qubit'
try:
    qml.device('lightning.gpu', wires=1)
    LOCAL_Q_DEV = 'lightning.gpu'
    print("PennyLane Compute Device: lightning.gpu (CUDA Accelerated)")
except Exception:
    print("PennyLane Compute Device: default.qubit (CPU)")
    print("  -> Run `pip install pennylane-lightning[gpu]` to enable GPU quantum simulation!")

dev = qml.device(LOCAL_Q_DEV, wires=N_QUBITS)

print(f'PennyLane {qml.__version__}  |  PyTorch {torch.__version__}')


def main():
    # ==========================================
    # 1 - Load Data for Both Tasks
    # ==========================================
    print("\n--- Loading Task 1A: Wildfire Classification Data ---")

    wf = pd.read_csv('data/wildfire_weather_daily.csv')
    wf['DATE'] = pd.to_datetime(wf['DATE'], errors='coerce')
    wf = wf[(wf['DATE'].dt.year >= 2018) & (wf['DATE'].dt.year <= 2021)].copy()

    TASK1_FEATURES = ['PRECIPITATION', 'MAX_TEMP', 'MIN_TEMP', 'AVG_WIND_SPEED',
                      'TEMP_RANGE', 'WIND_TEMP_RATIO', 'MONTH', 'DAY_OF_YEAR',
                      'LAGGED_PRECIPITATION', 'LAGGED_AVG_WIND_SPEED']
    wf = wf.dropna(subset=TASK1_FEATURES + ['FIRE_START_DAY'])

    X1 = wf[TASK1_FEATURES].values
    y1 = wf['FIRE_START_DAY'].astype(int).values
    years = wf['DATE'].dt.year.values

    train1 = years < 2021
    X1_tr, X1_te = X1[train1], X1[~train1]
    y1_tr, y1_te = y1[train1], y1[~train1]

    sc1 = StandardScaler()
    X1_tr_s = sc1.fit_transform(X1_tr)
    X1_te_s = sc1.transform(X1_te)

    pca1 = PCA(n_components=N_QUBITS)
    X1_tr_q = pca1.fit_transform(X1_tr_s)
    X1_te_q = pca1.transform(X1_te_s)

    mm1 = MinMaxScaler(feature_range=(0, np.pi))
    X1_tr_q = mm1.fit_transform(X1_tr_q)
    X1_te_q = mm1.transform(X1_te_q)

    print(f'Task 1A - Train: {X1_tr_q.shape}, Test: {X1_te_q.shape}')
    print(f'  PCA variance explained: {pca1.explained_variance_ratio_.sum():.1%}')
    print(f'  Fire days - train: {y1_tr.sum()}/{len(y1_tr)}, test: {y1_te.sum()}/{len(y1_te)}')

    print("\n--- Loading Task 2: Insurance Premium Regression Data ---")

    def find_col(df, keywords):
        kw = [k.lower() for k in keywords]
        for c in df.columns:
            if all(k in c.lower() for k in kw):
                return c
        return None

    def load_ho(path, years_list):
        frames = []
        for yr in years_list:
            df = pd.read_excel(path, sheet_name=f'{yr}HO', header=1, engine='openpyxl')
            df.columns = [c.replace('\n', ' ').strip() for c in df.columns]
            df = df.rename(columns={df.columns[0]: 'ZIP_Code'})
            df['ZIP_Code'] = pd.to_numeric(df['ZIP_Code'], errors='coerce')
            df = df.dropna(subset=['ZIP_Code'])
            df['ZIP_Code'] = df['ZIP_Code'].astype(int)
            df['Year'] = yr
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    ins1 = load_ho('data/insurance_2018_2019.XLS', [2018, 2019])
    ins2 = load_ho('data/insurance_2020_2021.XLS', [2020, 2021])
    common = list(set(ins1.columns) & set(ins2.columns))
    ins = pd.concat([ins1[common], ins2[common]], ignore_index=True)

    col_prem = find_col(ins, ['earned', 'premium'])
    col_exp  = find_col(ins, ['earned', 'exposure'])
    col_risk = find_col(ins, ['fire', 'risk', 'score'])

    # Match notebook: fire + smoke loss columns
    loss_cols  = [c for c in ins.columns if 'fire'  in c.lower() and 'incurred' in c.lower()]
    smoke_cols = [c for c in ins.columns if 'smoke' in c.lower() and 'incurred' in c.lower()]
    ins['total_fire_loss']    = ins[loss_cols].fillna(0).sum(axis=1)
    ins['total_smoke_loss']   = ins[smoke_cols].fillna(0).sum(axis=1)
    ins['total_insured_loss'] = ins['total_fire_loss'] + ins['total_smoke_loss']
    ins['prem_per_pol']       = ins[col_prem] / ins[col_exp].replace(0, np.nan)
    ins['loss_ratio']         = ins['total_insured_loss'] / ins[col_prem].replace(0, np.nan)

    # Match notebook T2_FEATURES
    T2_FEATURES = [col_risk, col_exp, 'total_insured_loss']
    ins2_clean = ins.dropna(subset=T2_FEATURES + ['Year']).copy()
    ins2_clean = ins2_clean.sort_values(['ZIP_Code', 'Year'])
    ins2_clean['next_prem'] = ins2_clean.groupby('ZIP_Code')[col_prem].shift(-1)
    ins2_clean = ins2_clean.dropna(subset=['next_prem'])

    X2 = ins2_clean[T2_FEATURES].values
    y2_raw = ins2_clean['next_prem'].values
    y2 = np.log1p(np.abs(y2_raw)) * np.sign(y2_raw)

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

    print(f'Task 2 - Train: {X2_tr_q.shape}, Test: {X2_te_q.shape}')
    print(f'  PCA variance: {pca2.explained_variance_ratio_.sum():.1%}')

    # ==========================================
    # Section A - Quantum Reservoir Computing
    # ==========================================
    print("\n--- Section A: Quantum Reservoir Computing (QRC) ---")
    N_RESERVOIR_LAYERS = 15
    n_qubits = N_QUBITS

    np.random.seed(SEED)
    reservoir_params = np.random.uniform(0, 2*np.pi, (N_RESERVOIR_LAYERS, n_qubits, 3))
    dev_qrc = qml.device(LOCAL_Q_DEV, wires=n_qubits)

    @qml.qnode(dev_qrc, interface='numpy')
    def reservoir_circuit(x, params):
        for i in range(n_qubits):
            qml.RX(x[i % len(x)], wires=i)
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
            features.append(reservoir_circuit(x_padded, params))
            if (idx+1) % 100 == 0:
                print(f'  {batch_label} {idx+1}/{n}  ({time.time()-t0:.1f}s)', end='\r')
        print(f'  {batch_label} {n}/{n} done in {time.time()-t0:.1f}s')
        return np.array(features)

    N_TRAIN_1A = min(500, len(X1_tr_q))
    N_TEST_1A  = min(200, len(X1_te_q))

    t0 = time.time()
    qrc_feat_tr_1 = extract_reservoir_features(X1_tr_q[:N_TRAIN_1A], reservoir_params, 'Train-1A')
    qrc_feat_te_1 = extract_reservoir_features(X1_te_q[:N_TEST_1A],  reservoir_params, 'Test-1A')
    qrc_time_1a = time.time() - t0

    clf_qrc = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
    clf_qrc.fit(qrc_feat_tr_1, y1_tr[:N_TRAIN_1A])
    y1_pred_qrc = clf_qrc.predict(qrc_feat_te_1)
    y1_prob_qrc = clf_qrc.predict_proba(qrc_feat_te_1)[:, 1]
    f1_qrc  = f1_score(y1_te[:N_TEST_1A], y1_pred_qrc)
    auc_qrc = roc_auc_score(y1_te[:N_TEST_1A], y1_prob_qrc)
    print(f'\n=== QRC - Task 1A ===\n  F1: {f1_qrc:.4f}  AUC: {auc_qrc:.4f}  |  Time: {qrc_time_1a:.1f}s')
    print(classification_report(y1_te[:N_TEST_1A], y1_pred_qrc, target_names=['No Fire', 'Fire']))

    N_TRAIN_2 = min(400, len(X2_tr_q))
    N_TEST_2  = min(100, len(X2_te_q))

    t0 = time.time()
    qrc_feat_tr_2 = extract_reservoir_features(X2_tr_q[:N_TRAIN_2], reservoir_params, 'Train-T2')
    qrc_feat_te_2 = extract_reservoir_features(X2_te_q[:N_TEST_2],  reservoir_params, 'Test-T2')
    qrc_time_t2 = time.time() - t0

    reg_qrc = Ridge(alpha=1.0)
    reg_qrc.fit(qrc_feat_tr_2, y2_tr[:N_TRAIN_2])
    y2_pred_qrc = reg_qrc.predict(qrc_feat_te_2)
    r2_qrc   = r2_score(y2_te[:N_TEST_2], y2_pred_qrc)
    rmse_qrc = np.sqrt(mean_squared_error(y2_te[:N_TEST_2], y2_pred_qrc))
    print(f'\n=== QRC - Task 2 ===\n  R2: {r2_qrc:.4f}  RMSE: {rmse_qrc:.4f}  |  Time: {qrc_time_t2:.1f}s')

    # ==========================================
    # Section B - Trainable Quantum Kernel
    # ==========================================
    print("\n--- Section B: Trainable Quantum Kernel ---")
    n_kernel_qubits = min(N_QUBITS, 6)
    dev_kernel = qml.device(LOCAL_Q_DEV, wires=n_kernel_qubits)

    @qml.qnode(dev_kernel, interface='autograd')
    def kernel_circuit(x1, x2, params):
        for i in range(n_kernel_qubits):
            qml.RY(x1[i % len(x1)] * params[i], wires=i)
            qml.RZ(x1[i % len(x1)] * params[n_kernel_qubits + i], wires=i)
        for i in range(n_kernel_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        for i in reversed(range(n_kernel_qubits - 1)):
            qml.CNOT(wires=[i, i+1])
        for i in reversed(range(n_kernel_qubits)):
            qml.RZ(-x2[i % len(x2)] * params[n_kernel_qubits + i], wires=i)
            qml.RY(-x2[i % len(x2)] * params[i], wires=i)
        return qml.probs(wires=range(n_kernel_qubits))

    def quantum_kernel(x1, x2, params):
        return kernel_circuit(x1, x2, params)[0]

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
                    row.append(pnp.array(0.0))
            K_rows.append(pnp.stack(row))
        K = pnp.stack(K_rows)
        K = K + K.T - pnp.diag(pnp.diag(K))
        return K

    def kernel_target_alignment(K, y):
        y = pnp.array(y, dtype=float)
        y_outer = pnp.outer(y, y)
        return pnp.sum(K * y_outer) / (
            pnp.sqrt(pnp.sum(K * K)) * pnp.sqrt(pnp.sum(y_outer * y_outer))
        )

    N_KERNEL    = min(120, len(X1_tr_q))
    N_KERNEL_TE = min(60,  len(X1_te_q))

    X_k_tr = pnp.array(X1_tr_q[:N_KERNEL, :n_kernel_qubits], requires_grad=False)
    y_k_tr = y1_tr[:N_KERNEL].copy()
    y_k_tr_signed = 2 * y_k_tr - 1
    X_k_te = pnp.array(X1_te_q[:N_KERNEL_TE, :n_kernel_qubits], requires_grad=False)
    y_k_te = y1_te[:N_KERNEL_TE]

    params_k = pnp.array(np.random.uniform(0.5, 1.5, 2 * n_kernel_qubits), requires_grad=True)
    opt_k = qml.AdamOptimizer(stepsize=0.1)
    n_kernel_epochs = 15

    t0 = time.time()
    for epoch in range(n_kernel_epochs):
        def cost_fn(p):
            K = kernel_matrix(X_k_tr, p)
            return -kernel_target_alignment(K, y_k_tr_signed)
        params_k, cost = opt_k.step_and_cost(cost_fn, params_k)
        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1:3d}  |  alignment = {-cost:.4f}  |  {time.time()-t0:.1f}s')
    kernel_train_time = time.time() - t0

    K_train = np.zeros((N_KERNEL, N_KERNEL))
    for i in range(N_KERNEL):
        for j in range(i, N_KERNEL):
            k = float(quantum_kernel(X_k_tr[i], X_k_tr[j], params_k))
            K_train[i, j] = K_train[j, i] = k

    K_test = np.zeros((N_KERNEL_TE, N_KERNEL))
    for i in range(N_KERNEL_TE):
        for j in range(N_KERNEL):
            K_test[i, j] = float(quantum_kernel(X_k_te[i], X_k_tr[j], params_k))

    svm_qk = SVC(kernel='precomputed', class_weight='balanced', probability=True, random_state=SEED)
    if len(np.unique(y_k_tr)) > 1:
        svm_qk.fit(K_train, y_k_tr)
        y_pred_qk = svm_qk.predict(K_test)
        y_prob_qk = svm_qk.predict_proba(K_test)[:, 1]
        f1_qk  = f1_score(y_k_te, y_pred_qk)
        auc_qk = roc_auc_score(y_k_te, y_prob_qk)
    else:
        f1_qk = auc_qk = 0.0
    print(f'\n=== Trainable Quantum Kernel - Task 1A ===\n  F1: {f1_qk:.4f}  AUC: {auc_qk:.4f}  |  Time: {kernel_train_time:.1f}s')

    # ==========================================
    # Section C - Quantum LSTM (QLSTM)
    # ==========================================
    print("\n--- Section C: Quantum LSTM (QLSTM) ---")
    n_qlstm_qubits = 4
    dev_qlstm = qml.device(LOCAL_Q_DEV, wires=n_qlstm_qubits)

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
            self.proj_in  = nn.Linear(input_size + hidden_size, n_qubits)
            self.q_forget = qml.qnn.TorchLayer(qlstm_circuit, weight_shapes)
            self.q_input  = qml.qnn.TorchLayer(qlstm_circuit, weight_shapes)
            self.q_cell   = qml.qnn.TorchLayer(qlstm_circuit, weight_shapes)
            self.q_output = qml.qnn.TorchLayer(qlstm_circuit, weight_shapes)
            self.proj_out = nn.Linear(n_qubits, hidden_size)

        def forward(self, x, states):
            h, c = states
            q_in  = torch.sigmoid(self.proj_in(torch.cat([x, h], dim=-1)))
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
            self.fc   = nn.Linear(hidden_size, output_size)

        def forward(self, x_seq):
            b = x_seq.size(0)
            h = torch.zeros(b, self.hidden_size).to(x_seq.device)
            c = torch.zeros(b, self.hidden_size).to(x_seq.device)
            for t in range(x_seq.size(1)):
                h, c = self.cell(x_seq[:, t, :], (h, c))
            return self.fc(h)

    # Build sequences using T2_FEATURES (matches notebook)
    ins_ts = ins.dropna(subset=T2_FEATURES + ['Year']).sort_values(['ZIP_Code', 'Year']).copy()
    sc_ts = StandardScaler()
    ins_ts[T2_FEATURES] = sc_ts.fit_transform(ins_ts[T2_FEATURES])

    sequences, targets = [], []
    for _, grp in ins_ts.groupby('ZIP_Code'):
        grp = grp.sort_values('Year')
        if len(grp) < 3:
            continue
        feats = grp[T2_FEATURES].values
        prems = grp[col_prem].values
        for i in range(len(grp) - 2):
            sequences.append(feats[i:i+2])
            targets.append(np.log1p(np.abs(prems[i+2])) * np.sign(prems[i+2]))

    sequences = np.array(sequences, dtype=np.float32)
    targets   = np.array(targets,   dtype=np.float32)

    split_idx = int(0.8 * len(sequences))
    X_ts_tr, X_ts_te = sequences[:split_idx], sequences[split_idx:]
    y_ts_tr, y_ts_te = targets[:split_idx],   targets[split_idx:]

    n_ts_features = X_ts_tr.shape[-1]
    n_ts_pca = min(n_qlstm_qubits, n_ts_features)

    pca_ts = PCA(n_components=n_ts_pca)
    pca_ts.fit(X_ts_tr.reshape(-1, n_ts_features))
    X_ts_tr_q = np.array([pca_ts.transform(s) for s in X_ts_tr], dtype=np.float32)
    X_ts_te_q = np.array([pca_ts.transform(s) for s in X_ts_te], dtype=np.float32)

    mm_ts = MinMaxScaler(feature_range=(0, np.pi))
    X_ts_tr_q = mm_ts.fit_transform(X_ts_tr_q.reshape(-1, n_ts_pca)).reshape(X_ts_tr_q.shape)
    X_ts_te_q = mm_ts.transform(X_ts_te_q.reshape(-1, n_ts_pca)).reshape(X_ts_te_q.shape)

    HIDDEN = 8
    QLSTM_EPOCHS = 30
    N_TS_TRAIN   = min(200, len(X_ts_tr_q))

    model_qlstm = QLSTMModel(n_ts_pca, HIDDEN, n_qlstm_qubits).to(device)
    optimizer_qlstm = torch.optim.Adam(model_qlstm.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_ts_tr_q[:N_TS_TRAIN])
    y_t = torch.tensor(y_ts_tr[:N_TS_TRAIN]).unsqueeze(-1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

    t0 = time.time()
    for epoch in range(QLSTM_EPOCHS):
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer_qlstm.zero_grad()
            loss = criterion(model_qlstm(xb), yb)
            loss.backward()
            optimizer_qlstm.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1:3d}  |  loss = {epoch_loss/len(loader):.4f}  |  {time.time()-t0:.1f}s')
    qlstm_time = time.time() - t0

    model_qlstm.eval()
    with torch.no_grad():
        y_pred_qlstm = model_qlstm(torch.tensor(X_ts_te_q).to(device)).cpu().numpy().flatten()
    r2_qlstm   = r2_score(y_ts_te, y_pred_qlstm)
    rmse_qlstm = np.sqrt(mean_squared_error(y_ts_te, y_pred_qlstm))
    print(f'\n=== QLSTM - Task 2 ===\n  R2: {r2_qlstm:.4f}  RMSE: {rmse_qlstm:.4f}  |  Time: {qlstm_time:.1f}s')

    # ==========================================
    # Section D - Quantum Transfer Learning
    # ==========================================
    print("\n--- Section D: Quantum Transfer Learning ---")
    n_tl_qubits = 4
    dev_tl = qml.device(LOCAL_Q_DEV, wires=n_tl_qubits)

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
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, n_qubits), nn.Sigmoid()
            )
            self.scale  = nn.Parameter(torch.tensor(float(np.pi)))
            self.quantum = qml.qnn.TorchLayer(transfer_circuit, tl_weight_shapes)
            self.fc_out  = nn.Linear(1, 1)

        def forward(self, x):
            h = self.classical(x) * self.scale
            q = self.quantum(h).unsqueeze(-1)
            return torch.sigmoid(self.fc_out(q)).squeeze(-1)

    N_TL_TRAIN = min(500, len(X1_tr_s))
    TL_EPOCHS  = 30

    X_tl_tr = torch.tensor(X1_tr_s[:N_TL_TRAIN].astype(np.float32)).to(device)
    y_tl_tr = torch.tensor(y1_tr[:N_TL_TRAIN].astype(np.float32)).to(device)
    X_tl_te = torch.tensor(X1_te_s.astype(np.float32)).to(device)

    model_tl = TransferModel(X_tl_tr.shape[1], hidden=32, n_qubits=n_tl_qubits).to(device)
    opt_tl = torch.optim.Adam(model_tl.parameters(), lr=0.005)
    criterion_tl = nn.BCELoss()

    t0 = time.time()
    for epoch in range(TL_EPOCHS):
        model_tl.train()
        opt_tl.zero_grad()
        loss = criterion_tl(model_tl(X_tl_tr), y_tl_tr)
        loss.backward()
        opt_tl.step()
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1:3d}  |  loss = {loss.item():.4f}  |  {time.time()-t0:.1f}s')
    tl_time = time.time() - t0

    model_tl.eval()
    with torch.no_grad():
        y_prob_tl = model_tl(X_tl_te).cpu().numpy()
    y_pred_tl = (y_prob_tl > 0.5).astype(int)
    f1_tl  = f1_score(y1_te, y_pred_tl)
    auc_tl = roc_auc_score(y1_te, y_prob_tl)
    print(f'\n=== Transfer Learning - Task 1A ===\n  F1: {f1_tl:.4f}  AUC: {auc_tl:.4f}  |  Time: {tl_time:.1f}s')
    print(classification_report(y1_te, y_pred_tl, target_names=['No Fire', 'Fire']))

    # ==========================================
    # Section F - Results Comparison
    # ==========================================
    print("\n=== COMPREHENSIVE RESULTS COMPARISON ===")

    clf_rf = RandomForestClassifier(200, class_weight='balanced', random_state=SEED)
    clf_rf.fit(X1_tr_s[:N_TRAIN_1A], y1_tr[:N_TRAIN_1A])
    f1_rf = f1_score(y1_te[:N_TEST_1A], clf_rf.predict(X1_te_s[:N_TEST_1A]))

    clf_xgb = xgb.XGBClassifier(n_estimators=200, eval_metric='logloss', random_state=SEED)
    clf_xgb.fit(X1_tr_s[:N_TRAIN_1A], y1_tr[:N_TRAIN_1A])
    f1_xgb = f1_score(y1_te[:N_TEST_1A], clf_xgb.predict(X1_te_s[:N_TEST_1A]))

    reg_ridge = Ridge(alpha=1.0)
    reg_ridge.fit(X2_tr_s[:N_TRAIN_2], y2_tr[:N_TRAIN_2])
    r2_ridge = r2_score(y2_te[:N_TEST_2], reg_ridge.predict(X2_te_s[:N_TEST_2]))

    print(f'\nTASK 1A — Wildfire Classification (F1)')
    print(f'  {"Model":<28} {"Type":<10} {"F1":>8}')
    print(f'  {"-"*50}')
    print(f'  {"Random Forest":<28} {"Classical":<10} {f1_rf:>8.4f}')
    print(f'  {"XGBoost":<28} {"Classical":<10} {f1_xgb:>8.4f}')
    print(f'  {"QRC + LogReg":<28} {"Quantum":<10} {f1_qrc:>8.4f}')
    print(f'  {"Trainable Kernel + SVM":<28} {"Quantum":<10} {f1_qk:>8.4f}')
    print(f'  {"Quantum Transfer Learning":<28} {"Hybrid":<10} {f1_tl:>8.4f}')

    print(f'\nTASK 2 — Premium Regression (R2, log-scale)')
    print(f'  {"Model":<28} {"Type":<10} {"R2":>8}')
    print(f'  {"-"*50}')
    print(f'  {"Ridge":<28} {"Classical":<10} {r2_ridge:>8.4f}')
    print(f'  {"QRC + Ridge":<28} {"Quantum":<10} {r2_qrc:>8.4f}')
    print(f'  {"QLSTM":<28} {"Quantum":<10} {r2_qlstm:>8.4f}')

    os.makedirs('results', exist_ok=True)
    results = {
        'task_1a': {
            'random_forest':      {'f1': round(float(f1_rf),   4), 'type': 'classical'},
            'xgboost':            {'f1': round(float(f1_xgb),  4), 'type': 'classical'},
            'qrc':                {'f1': round(float(f1_qrc),  4), 'qubits': N_QUBITS,        'type': 'quantum'},
            'trainable_kernel':   {'f1': round(float(f1_qk),   4), 'qubits': n_kernel_qubits, 'type': 'quantum'},
            'transfer_learning':  {'f1': round(float(f1_tl),   4), 'qubits': n_tl_qubits,     'type': 'hybrid'},
        },
        'task_2': {
            'ridge': {'r2': round(float(r2_ridge),  4), 'type': 'classical'},
            'qrc':   {'r2': round(float(r2_qrc),    4), 'qubits': N_QUBITS,       'type': 'quantum'},
            'qlstm': {'r2': round(float(r2_qlstm),  4), 'qubits': n_qlstm_qubits, 'type': 'quantum'},
        }
    }
    with open('results/advanced_quantum_results.json', 'w') as f:
        _json.dump(results, f, indent=2)

    print('\nFinal Results:')
    print(_json.dumps(results, indent=2))
    print('Saved to results/advanced_quantum_results.json')


if __name__ == "__main__":
    main()
