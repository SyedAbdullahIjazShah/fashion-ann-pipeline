import os
import yaml
import numpy as np
import pandas as pd
import h5py

def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params["train"]

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def train_tf(x_train, y_train, x_val, y_val, params):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.2)
    lr = params.get("learning_rate", 0.001)
    epochs = params.get("epochs", 10)
    batch_size = params.get("batch_size", 64)

    print("Building Sequential ANN Architecture via TensorFlow...")
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

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    model.save("models/model.h5")
    pd.DataFrame(history.history).to_csv("models/history.csv", index=False)

def train_numpy(x_train, y_train, x_val, y_val, params):
    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.2)
    lr = params.get("learning_rate", 0.001)
    epochs = params.get("epochs", 10)
    batch_size = params.get("batch_size", 64)

    print(f"Building Sequential ANN: Flatten(784) -> Dense({dense_units}, ReLU) -> Dropout({dropout_rate}) -> Dense(10, Softmax)")
    print(f"Compiling with Adam(lr={lr}) and sparse_categorical_crossentropy...")

    np.random.seed(42)
    # He normal for layer 1, Xavier for layer 2
    w1 = np.random.randn(784, dense_units).astype(np.float32) * np.sqrt(2.0 / 784.0)
    b1 = np.zeros(dense_units, dtype=np.float32)
    w2 = np.random.randn(dense_units, 10).astype(np.float32) * np.sqrt(2.0 / (dense_units + 10))
    b2 = np.zeros(10, dtype=np.float32)

    # Adam moments
    mw1, vw1 = np.zeros_like(w1), np.zeros_like(w1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mw2, vw2 = np.zeros_like(w2), np.zeros_like(w2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    x_train_flat = x_train.reshape(-1, 784).astype(np.float32)
    x_val_flat = x_val.reshape(-1, 784).astype(np.float32)
    n_samples = x_train_flat.shape[0]

    history_records = []

    print("\nTraining Sequential ANN Model...")
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(n_samples)
        x_shuffled = x_train_flat[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        correct_train = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            xb = x_shuffled[start_idx:end_idx]
            yb = y_shuffled[start_idx:end_idx]
            bs = xb.shape[0]

            # Forward
            z1 = np.dot(xb, w1) + b1
            a1 = np.maximum(0.0, z1)
            # Inverted Dropout
            mask = (np.random.rand(*a1.shape) >= dropout_rate).astype(np.float32) / (1.0 - dropout_rate)
            a1_drop = a1 * mask
            z2 = np.dot(a1_drop, w2) + b2
            probs = softmax(z2)

            # Metrics
            pred = np.argmax(probs, axis=1)
            correct_train += np.sum(pred == yb)
            loss_batch = -np.mean(np.log(probs[np.arange(bs), yb] + 1e-12))
            epoch_loss += loss_batch * bs

            # Backward
            dz2 = probs.copy()
            dz2[np.arange(bs), yb] -= 1.0
            dz2 /= bs

            dw2 = np.dot(a1_drop.T, dz2)
            db2 = np.sum(dz2, axis=0)

            da1 = np.dot(dz2, w2.T) * mask
            dz1 = da1 * (z1 > 0.0)

            dw1 = np.dot(xb.T, dz1)
            db1 = np.sum(dz1, axis=0)

            # Adam update
            t += 1
            for param, grad, m, v in [
                (w1, dw1, mw1, vw1), (b1, db1, mb1, vb1),
                (w2, dw2, mw2, vw2), (b2, db2, mb2, vb2)
            ]:
                m[:] = beta1 * m + (1.0 - beta1) * grad
                v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

        train_loss = epoch_loss / n_samples
        train_acc = correct_train / n_samples

        # Validation (Inference mode: no dropout)
        val_z1 = np.dot(x_val_flat, w1) + b1
        val_a1 = np.maximum(0.0, val_z1)
        val_z2 = np.dot(val_a1, w2) + b2
        val_probs = softmax(val_z2)
        val_preds = np.argmax(val_probs, axis=1)
        val_loss = -np.mean(np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-12))
        val_acc = np.mean(val_preds == y_val)

        print(f"Epoch {epoch:2d}/{epochs:2d} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
        history_records.append({
            "loss": float(train_loss),
            "accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc)
        })

    # Save to models/model.h5
    with h5py.File("models/model.h5", "w") as f:
        f.attrs["architecture"] = "Sequential: Flatten -> Dense(ReLU) -> Dropout -> Dense(Softmax)"
        f.attrs["dense_units"] = dense_units
        f.attrs["dropout_rate"] = dropout_rate
        f.create_dataset("weights/w1", data=w1)
        f.create_dataset("weights/b1", data=b1)
        f.create_dataset("weights/w2", data=w2)
        f.create_dataset("weights/b2", data=b2)

    pd.DataFrame(history_records).to_csv("models/history.csv", index=False)
    print("Saved trained model to models/model.h5")
    print("Saved training history to models/history.csv")

def main():
    print("=== [Stage 3: Train ANN Model] ===")
    os.makedirs("models", exist_ok=True)
    params = load_params()

    print("Loading processed training and validation sets...")
    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    try:
        import tensorflow as tf
        train_tf(x_train, y_train, x_val, y_val, params)
    except Exception as e:
        print(f"TensorFlow runtime unavailable ({e}). Using native high-performance ANN engine...")
        train_numpy(x_train, y_train, x_val, y_val, params)

if __name__ == "__main__":
    main()
