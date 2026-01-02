from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from preprocess import load_nsl_kdd, prepare_autoencoder_data

# Load and preprocess data
df = load_nsl_kdd("data/NSL-KDD/KDDTrain+.txt")
X, scaler = prepare_autoencoder_data(df)

# Autoencoder architecture
input_dim = X.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation="relu")(input_layer)
encoded = Dense(16, activation="relu")(encoded)

decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train model
autoencoder.fit(
    X, X,
    epochs=20,
    batch_size=256,
    validation_split=0.1
)

# Save model
autoencoder.save("model.h5")
print("✅ Autoencoder trained and saved as model.h5")
