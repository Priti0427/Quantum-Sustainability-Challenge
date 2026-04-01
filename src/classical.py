from sklearn.metrics import accuracy_score
from .gpu import GPU_AVAILABLE, RandomForestClassifier, to_device, to_numpy
import numpy as np

def train_rf(X, y):
    model = RandomForestClassifier(n_estimators=200)
    model.fit(to_device(X), to_device(y))
    return model

def evaluate(model, X, y):
    preds = to_numpy(model.predict(to_device(X)))
    acc = accuracy_score(np.asarray(y), np.asarray(preds))
    print(f"RF Accuracy: {acc}  (gpu={GPU_AVAILABLE})")
