# Fashion-MNIST ANN Pipeline with Git & DVC

An end-to-end reproducible Machine Learning pipeline utilizing **TensorFlow**, **Git**, and **Data Version Control (DVC)** with **Google Drive** as remote artifact storage.

## Objectives
- **Model:** Fully-connected Artificial Neural Network (ANN) classifying 70,000 $28 \times 28$ grayscale Fashion-MNIST images into 10 clothing categories.
- **Accuracy Target:** $\ge 85\%$ test accuracy.
- **Reproducibility:** Pipeline managed and executed via `dvc repro`.
- **Versioning:** Code tracked in Git, large datasets and model weights tracked via DVC pointer files.

## Project Structure
```
fashion-ann-pipeline/
├── .dvc/                   # DVC configuration and tracking
├── data/
│   ├── raw/               # Downloaded Fashion-MNIST arrays (.npz)
│   └── processed/         # Normalized and train/val/test splits
├── models/
│   ├── model.h5           # Trained ANN model
│   └── history.csv        # Training epochs loss and accuracy log
├── src/
│   ├── prepare.py         # Stage 1: Data ingestion
│   ├── preprocess.py      # Stage 2: Feature normalization and splitting
│   ├── train.py           # Stage 3: Sequential ANN model training
│   └── evaluate.py        # Stage 4: Test evaluation and confusion matrix
├── params.yaml            # Central hyperparameters
├── dvc.yaml               # Pipeline stage definitions
└── metrics.json           # Evaluation metrics output
```
