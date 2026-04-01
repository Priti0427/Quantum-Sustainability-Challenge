def create_features(df):
    df["month"] = df["date"].dt.month

    for col in df.select_dtypes(include="number").columns:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_roll3"] = df[col].rolling(3).mean()

    return df.dropna()
