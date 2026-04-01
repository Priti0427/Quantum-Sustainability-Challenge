"""
quantum.py — Core Quantum ML architectures.

Includes:
- Transfer Learning (Hybrid Classical-Quantum)
- NEW: QMoE (Quantum Mixture of Experts for Climate Regimes)
- NEW: QART (Quantum Adaptive Reservoir Transformer)
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

from src.gpu import make_qdevice

SEED = 42
N_QUBITS = 8
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── shared: PCA + angle encoding ─────────────────────────────────────────────

def encode_for_quantum(X_tr, X_te, n_components):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    return X_tr_s, X_te_s

# ==========================================
# BASE ARCHITECTURE: Quantum Transfer Learning
# ==========================================

n_tl_qubits = 4
dev_tl = make_qdevice(n_tl_qubits)

@qml.qnode(dev_tl, interface='torch')
def _transfer_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_tl_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_tl_qubits))
    return qml.expval(qml.PauliZ(0))

_tl_weight_shapes = {'weights': (2, n_tl_qubits, 3)}

class TransferModel(nn.Module):
    def __init__(self, input_dim, hidden=32, n_qubits=4):
        super().__init__()
        self.classical = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_qubits), nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.tensor(np.pi))
        self.quantum = qml.qnn.TorchLayer(_transfer_circuit, _tl_weight_shapes)
        self.fc_out = nn.Linear(1, 1)

    def forward(self, x):
        h = self.classical(x) * self.scale
        q_out = self.quantum(h).unsqueeze(-1)
        return torch.sigmoid(self.fc_out(q_out)).squeeze(-1)

def run_transfer_learning(X_tr_s, X_te_s, y_tr, y_te, n_train=500, epochs=30, lr=0.005):
    """Quantum transfer learning classifier (Task 1A). Returns f1, auc, preds, probs."""
    import torch
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_t = torch.tensor(X_tr_s[:n_train].astype(np.float32)).to(device)
    y_t = torch.tensor(y_tr[:n_train].astype(np.float32)).to(device)
    X_te_t = torch.tensor(X_te_s.astype(np.float32)).to(device)

    model = TransferModel(X_t.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        probs = model(X_te_t).cpu().numpy()
    
    preds = (probs > 0.2).astype(int)
    f1 = f1_score(y_te, preds)
    auc = roc_auc_score(y_te, probs)
    return f1, auc, preds, probs


# ==========================================
# ADVANCED ARCHITECTURE 1: QMoE Fire
# ==========================================

class QuantumExpert(nn.Module):
    """A single quantum expert with a specific entanglement topology."""
    def __init__(self, n_qubits=4, n_layers=2, topology='ring'):
        super().__init__()
        self.n_qubits = n_qubits
        self.topology = topology
        self.dev = make_qdevice(n_qubits)
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        @qml.qnode(self.dev, interface='torch')
        def _expert_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            if topology == 'ring':
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            elif topology == 'star':
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            else:
                qml.SimplifiedTwoDesign(initial_layer_weights=weights[0,:,0], weights=weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        self.q_layer = qml.qnn.TorchLayer(_expert_circuit, self.weight_shapes)

class QMoE_Fire(nn.Module):
    """Quantum Mixture of Experts routed by Climate Regime."""
    def __init__(self, input_dim, n_experts=4, n_qubits=4):
        super().__init__()
        self.n_experts = n_experts
        
        self.router = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_experts),
            nn.Softmax(dim=-1)
        )
        
        topologies = ['ring', 'star', 'ladder', 'full']
        self.experts = nn.ModuleList([
            QuantumExpert(n_qubits=n_qubits, topology=topologies[i % 4]) 
            for i in range(n_experts)
        ])
        
        self.fc_out = nn.Linear(n_qubits, 1)

    def forward(self, x):
        gating_weights = self.router(x)
        expert_outputs = torch.stack([expert.q_layer(x[:, :expert.n_qubits]) for expert in self.experts], dim=1)
        fused_q_features = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return torch.sigmoid(self.fc_out(fused_q_features)).squeeze(-1)


# ==========================================
# ADVANCED ARCHITECTURE 2: QART
# ==========================================

class QART(nn.Module):
    """Quantum Adaptive Reservoir Transformer."""
    def __init__(self, seq_len, input_dim, n_qubits=4, hidden_dim=16):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = make_qdevice(n_qubits)
        
        self.proj = nn.Linear(input_dim, n_qubits)
        self.weight_shapes = {"weights": (2, n_qubits, 3)}
        
        @qml.qnode(self.dev, interface='torch')
        def _q_attention_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        self.q_transformer = qml.qnn.TorchLayer(_q_attention_circuit, self.weight_shapes)
        
        self.fc = nn.Sequential(
            nn.Linear(n_qubits * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_seq):
        batch_size, seq_len, _ = x_seq.size()
        q_features = []
        for t in range(seq_len):
            h_t = torch.sigmoid(self.proj(x_seq[:, t, :])) * np.pi
            q_out = self.q_transformer(h_t)
            q_features.append(q_out)
            
        fused_features = torch.cat(q_features, dim=1)
        return self.fc(fused_features)
