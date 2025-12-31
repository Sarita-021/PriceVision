import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.tfidf_ridge import build_tfidf_ridge
from utils.validation import create_validation_split
from metrics.smape import smape
from config import TARGET_COL, TEXT_COLS


# Load data
df = pd.read_csv("data/train.csv")

# Combine text fields
df["text"] = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)

# Create validation split
train_df, val_df = create_validation_split(df, TARGET_COL)

X_train = train_df["text"].values
X_val = val_df["text"].values

# Scaled Log-transform target
y_train = np.log1p(train_df[TARGET_COL].values) 
y_val = val_df[TARGET_COL].values  # keep original for SMAPE

# Build model
model = build_tfidf_ridge(
    max_features=50000,
    ngram_range=(1, 2),
    alpha=1.0
)

# Train
model.fit(X_train, y_train)

# Predict
val_pred_log = model.predict(X_val)
val_pred = np.expm1(model.predict(X_val)) 

# Sanity Check 1: Negative predictions
num_negative = (val_pred < 0).sum()
print(f"Negative predictions: {num_negative}")

# SMAPE safety clipping
low = np.percentile(train_df[TARGET_COL].values, 1)
high = np.percentile(train_df[TARGET_COL].values, 99)

val_pred = np.clip(val_pred, low, high)

# Post-prediction bias correction
bucket_medians = train_df.groupby("price_bucket")[TARGET_COL].median()

def bias_correct(row):
    b = row["price_bucket"]
    if b == 0:
        return row["pred"] * 0.85   # reduce over-prediction
    elif b == 4:
        return row["pred"] * 1.15   # increase under-prediction
    return row["pred"]

val_df["pred"] = val_pred
val_df["pred"] = val_df.apply(bias_correct, axis=1)
val_pred = val_df["pred"].values


# Evaluate
score = smape(y_val, val_pred)

# Attach predictions for analysis
val_df = val_df.copy()
val_df["pred"] = val_pred

print(f"Validation SMAPE (TF-IDF + Ridge): {score:.4f}")

# Attach predictions for analysis
val_df = val_df.copy()
val_df["pred"] = val_pred

print("\nBucket-wise SMAPE:")
for bucket in sorted(val_df["price_bucket"].unique()):
    mask = val_df["price_bucket"] == bucket
    b_score = smape(
        val_df.loc[mask, TARGET_COL],
        val_df.loc[mask, "pred"]
    )
    print(f"Bucket {bucket}: {b_score:.2f}")

# Bucket 0 (cheapest)
# Bucket 4 (expensive)