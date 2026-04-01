"""
gpu.py — single source of truth for GPU availability.

Import GPU_AVAILABLE and the aliased classes/functions from here.
Everything falls back to CPU transparently if RAPIDS / GPU is absent.
"""

# ── detect once ──────────────────────────────────────────────────────────────
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()          # raises if no CUDA device
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# ── array helpers ─────────────────────────────────────────────────────────────
if GPU_AVAILABLE:
    import cupy as cp
    def to_device(x):
        """Move a numpy array (or array-like) to GPU."""
        return cp.asarray(x)
    def to_numpy(x):
        """Move a cupy array back to CPU numpy."""
        return cp.asnumpy(x)
else:
    import numpy as np
    def to_device(x):
        return x
    def to_numpy(x):
        return x

# ── scikit-learn / cuML drop-ins ──────────────────────────────────────────────
if GPU_AVAILABLE:
    try:
        from cuml.ensemble   import RandomForestClassifier
        from cuml.linear_model import LinearRegression
        from cuml.preprocessing import StandardScaler
        from cuml.manifold   import UMAP
        from cuml.cluster    import HDBSCAN
    except ImportError:
        GPU_AVAILABLE = False          # cuML missing despite CUDA — fall back

if not GPU_AVAILABLE:
    from sklearn.ensemble    import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import umap as _umap
    import hdbscan as _hdbscan
    UMAP   = _umap.UMAP
    HDBSCAN = _hdbscan.HDBSCAN

# ── PennyLane device factory ──────────────────────────────────────────────────
def make_qdevice(n_wires: int):
    """Return the best available PennyLane device for n_wires qubits."""
    import pennylane as qml
    if GPU_AVAILABLE:
        try:
            return qml.device("lightning.gpu", wires=n_wires)
        except Exception:
            pass
        try:
            return qml.device("lightning.qubit", wires=n_wires)
        except Exception:
            pass
    return qml.device("default.qubit", wires=n_wires)
