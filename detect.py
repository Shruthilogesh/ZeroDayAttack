import os
import numpy as np
import pandas as pd
import tensorflow as tf
from ml.preprocess import load_nsl_kdd

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model", "autoencoder.keras")
DATA_PATH = os.path.join(BASE_DIR, "data", "NSL-KDD", "KDDTest+.txt")

model = tf.keras.models.load_model(MODEL_PATH)

df = load_nsl_kdd(DATA_PATH)

X = df.drop(columns=["label", "difficulty"], errors="ignore").astype("float32")

recon = model.predict(X, batch_size=256)
mse = np.mean(np.square(X - recon), axis=1)

def detect(threshold):
    results = []
    for i, score in enumerate(mse[:100]):  # LIMIT to avoid RAM crash
        results.append({
            "index": int(i),
            "score": float(score),
            "status": "Anomaly" if score > threshold else "Normal"
        })
    return results
