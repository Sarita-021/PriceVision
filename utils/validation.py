import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def add_price_buckets(df, target_col, n_buckets=5):
    """
    Adds quantile-based price buckets for stratified splitting.
    """
    df = df.copy()
    df["price_bucket"] = pd.qcut(
        df[target_col],
        q=n_buckets,
        labels=False,
        duplicates="drop"
    )
    return df


def create_validation_split(
    df,
    target_col,
    test_size=0.2,
    random_state=42
):
    """
    Creates bucket-aware train/validation split.
    """
    df = add_price_buckets(df, target_col)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["price_bucket"]
    )

    return train_df, val_df
