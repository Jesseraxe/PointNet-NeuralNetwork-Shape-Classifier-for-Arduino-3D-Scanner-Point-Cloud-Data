# 3D Shape Classifier using PointNet

This project implements and trains a PointNet model to classify 3D shapes (sphere, cone, cube, cylinder, cuboid) from point cloud data.

## Features
* **PointNet Model:** Implements the PointNet architecture for point cloud classification.
* **Data Handling:** Custom `Dataset` class for loading point cloud data from `.TXT` files, including normalization and point sampling.
* **Data Augmentation:** Includes jittering, scaling, and translation augmentation for training data.
* **Training:**
    * Trains the PointNet model using PyTorch.
    * Uses AdamW optimizer, learning rate scheduling (ReduceLROnPlateau), class weighting, gradient clipping, and early stopping.
    * Saves the best performing model based on validation accuracy.
    * Logs training progress and saves training curves (loss and accuracy).
* **Evaluation:**
    * Evaluates the trained model on a test set.
    * Generates and saves a classification report and confusion matrices (standard and normalized).
    * Saves evaluation metrics (accuracy, loss).

## Data Format
The script expects point cloud data organized as follows:
```
<data_dir>/
├── train/
│   ├── sphere/
│   │   ├── sphere_001.TXT
│   │   └── ...
│   ├── cone/
│   │   └── ...
│   ├── cube/
│   │   └── ...
│   ├── cylinder/
│   │   └── ...
│   └── cuboid/
│       └── ...
├── val/
│   └── (similar structure as train)
└── test/
    └── (similar structure as train)
```

* Each `.TXT` file should contain 3D point coordinates (X, Y, Z), typically comma-separated, with 3 header lines to skip.
* The script normalizes the point clouds and samples/pads them to a fixed number of points (default: 1024).

## Requirements
* Python 3.x
* NumPy
* PyTorch
* Scikit-learn
* Seaborn
* Matplotlib

You can install the required libraries using pip:
```bash
pip install numpy torch scikit-learn seaborn matplotlib
```

## Usage
Run the script from the command line:
```bash
python shape_classifier.py --mode <mode> --data_dir <path_to_data> [options]
```

### Arguments:
* `--mode`: train, eval, or both (default: both).
* `--data_dir`: Path to the root directory containing train, val, and test splits (default: data).
* `--num_epochs`: Number of training epochs (default: 100).
* `--batch_size`: Training batch size (default: 8).
* `--learning_rate`: Training learning rate (default: 0.0005).

### Examples:
Train and evaluate:
```bash
python shape_classifier.py --data_dir ./my_shape_data
```

Train only:
```bash
python shape_classifier.py --mode train --data_dir ./my_shape_data --num_epochs 50
```

Evaluate only (requires a trained model at assets/model/best_model.pth):
```bash
python shape_classifier.py --mode eval --data_dir ./my_shape_data
```

## Output Files
The script creates an assets directory containing:
* `assets/model/best_model.pth`: Saved state dictionary of the best trained model.
* `assets/training_metrics.json`: Metrics recorded during training.
* `assets/training_curves.png`: Plot of training/validation loss and accuracy.
* `assets/evaluation/classification_report.txt`: Classification report from the test set evaluation.
* `assets/evaluation/confusion_matrix.png`: Confusion matrix plot.
* `assets/evaluation/confusion_matrix_normalized.png`: Normalized confusion matrix plot.
