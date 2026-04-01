from .gpu import UMAP, HDBSCAN, to_device, to_numpy

def run_clustering(X):
    X_d  = to_device(X)
    emb  = UMAP().fit_transform(X_d)
    labels = HDBSCAN().fit_predict(emb)
    return to_numpy(emb), to_numpy(labels)
