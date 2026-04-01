import pandas as pd

def prepare_wildfire(df):
    df["date"] = pd.to_datetime(df["date"])
    df["fire"] = (df["FIRE_NAME"] != "no_fire").astype(int)
    return df
