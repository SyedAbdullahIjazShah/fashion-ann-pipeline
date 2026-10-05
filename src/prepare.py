import os
import gzip
import urllib.request
import numpy as np

def download_and_extract(url, offset):
    with urllib.request.urlopen(url) as r:
        with gzip.GzipFile(fileobj=r) as f:
            f.read(offset)
            return f.read()

def load_from_mirror():
    base_url = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    print("Downloading Fashion-MNIST directly from official Zalando repository mirror...")
    
    print("Fetching train images...")
    raw_x_train = download_and_extract(base_url + "train-images-idx3-ubyte.gz", 16)
    x_train = np.frombuffer(raw_x_train, dtype=np.uint8).reshape(-1, 28, 28)

    print("Fetching train labels...")
    raw_y_train = download_and_extract(base_url + "train-labels-idx1-ubyte.gz", 8)
    y_train = np.frombuffer(raw_y_train, dtype=np.uint8)

    print("Fetching test images...")
    raw_x_test = download_and_extract(base_url + "t10k-images-idx3-ubyte.gz", 16)
    x_test = np.frombuffer(raw_x_test, dtype=np.uint8).reshape(-1, 28, 28)

    print("Fetching test labels...")
    raw_y_test = download_and_extract(base_url + "t10k-labels-idx1-ubyte.gz", 8)
    y_test = np.frombuffer(raw_y_test, dtype=np.uint8)

    return (x_train, y_train), (x_test, y_test)

def main():
    print("=== [Stage 1: Prepare Data] ===")
    os.makedirs("data/raw", exist_ok=True)

    loaded = False
    try:
        import tensorflow as tf
        print("Loading Fashion-MNIST dataset from tf.keras.datasets...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        loaded = True
    except Exception as e:
        print(f"tf.keras.datasets load failed ({e}), falling back to direct mirror...")

    if not loaded:
        (x_train, y_train), (x_test, y_test) = load_from_mirror()

    print(f"Train samples: {x_train.shape[0]}, Shape: {x_train.shape[1:]}")
    print(f"Test samples:  {x_test.shape[0]}, Shape: {x_test.shape[1:]}")

    np.save("data/raw/x_train.npy", x_train)
    np.save("data/raw/y_train.npy", y_train)
    np.save("data/raw/x_test.npy", x_test)
    np.save("data/raw/y_test.npy", y_test)

    print("Saved raw dataset arrays to data/raw/ successfully.")

if __name__ == "__main__":
    main()

