import os
import numpy as np
import tensorflow as tf

def main():
    print("=== [Stage 1: Prepare Data] ===")
    os.makedirs("data/raw", exist_ok=True)

    print("Loading Fashion-MNIST dataset from tf.keras.datasets...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    print(f"Train samples: {x_train.shape[0]}, Shape: {x_train.shape[1:]}")
    print(f"Test samples:  {x_test.shape[0]}, Shape: {x_test.shape[1:]}")

    np.save("data/raw/x_train.npy", x_train)
    np.save("data/raw/y_train.npy", y_train)
    np.save("data/raw/x_test.npy", x_test)
    np.save("data/raw/y_test.npy", y_test)

    print("Saved raw dataset arrays to data/raw/ successfully.")

if __name__ == "__main__":
    main()
