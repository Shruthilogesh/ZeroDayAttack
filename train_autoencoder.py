import os
import pandas as pd
import tensorflow as tf
from preprocess import load_nsl_kdd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data/NSL-KDD/KDDTrain+.txt")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = load_nsl_kdd(DATA_PATH)

# Use ONLY normal traffic
df = df[df["label"] == "normal"]

X = df.drop(columns=["label", "difficulty"]).astype("float32")

# Save feature order
X.columns.to_series().to_csv(
    os.path.join(MODEL_DIR, "feature_columns.csv"),
    index=False,
    header=False
)

input_dim = X.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(input_dim, activation="sigmoid")
])

model.compile(optimizer="adam", loss="mse")

model.fit(X, X, epochs=15, batch_size=256, validation_split=0.1)

model.save(os.path.join(MODEL_DIR, "autoencoder.keras"))

print("✅ Model trained successfully")
