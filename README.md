# Extreme Event Prediction Pipeline

Python framework for predicting extreme events in turbulent systems using recurrence analysis and deep learning. Simulates combustion dynamics, extracts temporal patterns through phase space reconstruction, and classifies stability regimes with CNNs.

**Install dependencies:**
```bash
pip install numpy scipy matplotlib pandas tqdm scikit-learn torch torchvision
```

**Run the pipeline:**
```bash
python main_pipeline.py
```

## Pipeline Overview

The pipeline executes 6 modules sequentially:

1. **ROM_and_segmentation.py** - Simulates thermoacoustic combustion and segments pressure signals
2. **cao_theorem.py** - Determines optimal embedding dimensions using Cao's method
3. **average_mutual_information.py** - Calculates optimal time delays for phase space reconstruction
4. **recurrence_matrix_generation.py** - Generates 450×450 recurrence matrices
5. **classification.py** - Classifies stability regimes using chaos detection (0-1 test)
6. **convolutional_neural_network.py** - Trains CNN for automated classification

## Module Details

### 1. ROM & Segmentation
Simulates a thermoacoustic combustor with vortex-flame interactions across 30 flow velocities (7.0-10.0 m/s). Segments pressure time series into 500-point overlapping windows.

**Output:** `pressure_segments/`

### 2. Cao's Theorem
Applies false nearest neighbors to find optimal embedding dimension for each segment.

**Output:** `data_with_embed/` (filenames append `_embed_[dim]`)

### 3. Average Mutual Information
Computes AMI to find optimal time delay tau for phase space reconstruction.

**Output:** `data_with_embed_tau/` (filenames append `_tau_[value]`)

### 4. Recurrence Matrices
Performs time-delay embedding and computes pairwise distances to create binary recurrence matrices.

**Output:** `recurrence_matrices/` + visualization of first 50 matrices

### 5. Classification
Classifies matrices into 4 stability types using geometric analysis and 0-1 chaos test:
- **Regular:** periodic/quasi-periodic, K <= 0.2
- **Type II:** Type II intermittency, 2nd quadrant (F_b ~ 1.0, centroid top-left) **System  hovers near a stable oscillatory state but is perturbed into erratic behavior**
- **Chaotic:** K >= 0.8

Ambiguous cases are excluded.

**Output:** `classified_matrices/` + CSV with metrics + visualizations

### 6. CNN Training
Trains a convolutional neural network on classified matrices:
- Architecture: 3 conv layers (32->64->128 filters) + 2 fully connected layers
- Training: 50 epochs, batch size 16, 80/20 train/val split
- Outputs model weights, training curves, and confusion matrix

**Output:** `recurrence_matrix_cnn.pth` + plots

## Output Structure

```
project/
|--- pressure_segments/           # Raw pressure data segments
|--- data_with_embed/             # With embedding dimensions
|--- data_with_embed_tau/         # With time delays
|--- recurrence_matrices/         # 450×450 binary matrices
|--- classified_matrices/         # Labeled by stability type
|--- recurrence_matrix_cnn.pth    # Trained model
|--- training_curves.png
|--- confusion_matrix.png
```

## Manual Execution

Run modules individually if needed:
```bash
python ROM_and_segmentation.py
python cao_theorem.py
python average_mutual_information.py
python recurrence_matrix_generation.py
python classification.py
python convolutional_neural_network.py
```

## Requirements

- Python 3.7+
- NumPy, SciPy, Matplotlib, Pandas, tqdm, scikit-learn, PyTorch




