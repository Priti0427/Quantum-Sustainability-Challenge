from sklearn.model_selection import train_test_split
from .gpu import StandardScaler, to_device, to_numpy

def split_scale(df, target, test_size, random_state):
    # Drop non-numeric and identifier columns
    drop_cols = [target] + [c for c in ["date", "county", "FIRE_NAME"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target].values

    X = X.select_dtypes(include="number")

    scaler = StandardScaler()
    X_scaled = to_numpy(scaler.fit_transform(to_device(X.values)))

    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
