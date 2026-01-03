import pandas as pd
import numpy as np

TARGET = "isFraud"
DROP_COLS = ["TransactionID"]

def preprocess(df, training=True):
    df = df.copy()

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    y = None
    if training and TARGET in df.columns:
        y = df[TARGET]
        df.drop(columns=[TARGET], inplace=True)

    # ---- CATEGORICAL FEATURES ----
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna("missing")
        df[col] = df[col].astype("category")

    # ---- NUMERICAL FEATURES ----
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(-999)

    return df, y