import os
import yaml
import numpy as np

def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params["preprocess"]

def stratified_train_val_split(x, y, test_size, seed):
    try:
        from sklearn.model_selection import train_test_split
        return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)
    except Exception:
        np.random.seed(seed)
        train_idx, val_idx = [], []
        for cls in np.unique(y):
            cls_indices = np.where(y == cls)[0]
            shuffled = np.random.permutation(cls_indices)
            split_point = int(len(shuffled) * (1 - test_size))
            train_idx.extend(shuffled[:split_point])
            val_idx.extend(shuffled[split_point:])
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        return x[train_idx], x[val_idx], y[train_idx], y[val_idx]

def main():
    print("=== [Stage 2: Preprocess Data] ===")
    os.makedirs("data/processed", exist_ok=True)
    params = load_params()

    val_size = params.get("val_size", 0.15)
    seed = params.get("seed", 42)

    print("Loading raw data from data/raw/...")
    x_train_raw = np.load("data/raw/x_train.npy")
    y_train_raw = np.load("data/raw/y_train.npy")
    x_test_raw = np.load("data/raw/x_test.npy")
    y_test_raw = np.load("data/raw/y_test.npy")

    print("Normalizing pixel values to [0, 1] range...")
    x_train_norm = x_train_raw.astype("float32") / 255.0
    x_test_norm = x_test_raw.astype("float32") / 255.0

    print(f"Splitting validation set (val_size={val_size}, seed={seed})...")
    x_train, x_val, y_train, y_val = stratified_train_val_split(
        x_train_norm,
        y_train_raw,
        test_size=val_size,
        seed=seed
    )

    print(f"Final splits -> Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test: {x_test_norm.shape[0]}")

    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/x_test.npy", x_test_norm)
    np.save("data/processed/y_test.npy", y_test_raw)

    print("Saved processed arrays to data/processed/ successfully.")

if __name__ == "__main__":
    main()
