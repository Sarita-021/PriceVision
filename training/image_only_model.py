import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import TARGET_COL, TEXT_COLS
from utils.validation import create_validation_split
from metrics.smape import smape

X_img = np.load("image_embeddings.npy")
df = pd.read_csv("train_with_images.csv")

y = df["price"].values

# Create validation split
train_df, val_df = create_validation_split(df, TARGET_COL)

# Split image embeddings according to train/val indices
X_train_img = X_img[train_df.index]
X_val_img = X_img[val_df.index]

from sklearn.linear_model import Ridge

y_train = np.log1p(train_df["price"].values)
y_val = val_df["price"].values

model = Ridge(alpha=10.0)
model.fit(X_train_img, y_train)

val_pred = np.expm1(model.predict(X_val_img))

print("Image-only SMAPE:", smape(y_val, val_pred))
