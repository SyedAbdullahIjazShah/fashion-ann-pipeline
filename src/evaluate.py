import os
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def evaluate_model():
    print("=== [Stage 4: Evaluate Model] ===")
    print("Loading trained model and processed test set...")

    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    x_test_flat = x_test.reshape(-1, 784).astype(np.float32)

    evaluated = False
    test_loss = 0.0
    test_acc = 0.0
    y_pred = None

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("models/model.h5")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        y_pred_probs = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        evaluated = True
    except Exception:
        pass

    if not evaluated:
        # Load directly from HDF5
        with h5py.File("models/model.h5", "r") as f:
            w1 = f["weights/w1"][:]
            b1 = f["weights/b1"][:]
            w2 = f["weights/w2"][:]
            b2 = f["weights/b2"][:]

        z1 = np.dot(x_test_flat, w1) + b1
        a1 = np.maximum(0.0, z1)
        z2 = np.dot(a1, w2) + b2
        probs = softmax(z2)
        y_pred = np.argmax(probs, axis=1)

        test_loss = float(-np.mean(np.log(probs[np.arange(len(y_test)), y_test] + 1e-12)))
        test_acc = float(np.mean(y_pred == y_test))

    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Build Confusion Matrix
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1

    # Plot Confusion Matrix
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Fashion-MNIST ANN Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion matrix to models/confusion_matrix.png")

    metrics = {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_acc), 4)
    }

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics to metrics.json:", metrics)

if __name__ == "__main__":
    evaluate_model()
