import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def main():
    print("=== [Stage 4: Evaluate Model] ===")
    print("Loading trained model and processed test set...")
    model = tf.keras.models.load_model("models/model.h5")
    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    print("Evaluating test metrics...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Predict classes for confusion matrix
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Plot Confusion Matrix
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Fashion-MNIST ANN Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)

    # Label cells
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
    print("Saved confusion matrix image to models/confusion_matrix.png")

    # Write metrics.json
    metrics = {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_acc), 4)
    }

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics to metrics.json:", metrics)

if __name__ == "__main__":
    main()
