import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params["train"]

def main():
    print("=== [Stage 3: Train ANN Model] ===")
    os.makedirs("models", exist_ok=True)
    params = load_params()

    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.2)
    lr = params.get("learning_rate", 0.001)
    epochs = params.get("epochs", 10)
    batch_size = params.get("batch_size", 64)

    print(f"Hyperparameters: units={dense_units}, dropout={dropout_rate}, lr={lr}, epochs={epochs}, batch={batch_size}")

    print("Loading processed training and validation sets...")
    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    print("Building Sequential ANN Architecture (Flatten -> Dense -> Dropout -> Dense Softmax)...")
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    print("Training model...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    model_path = "models/model.h5"
    history_path = "models/history.csv"

    model.save(model_path)
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    print(f"Model saved to {model_path}")
    print(f"History saved to {history_path}")

if __name__ == "__main__":
    main()
