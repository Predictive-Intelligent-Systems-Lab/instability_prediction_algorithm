# Extreme Event Prediction Analysis Pipeline

A comprehensive Python framework for extreme event prediction through recurrence matrix generation, chaos detection, and deep learning classification.

---

## Overview

This pipeline implements a complete workflow for studying extreme event prediction in turbulent dynamical systems. It simulates combustion dynamics using a reduced-order model (ROM), extracts temporal patterns through phase space reconstruction, generates recurrence matrices, and classifies stability regimes using convolutional neural networks.

**What it does**: Simulates combustion pressure signals, identifies dynamical patterns, and classifies combustion stability regimes (regular, intermittent, chaotic).

**How it works**: Physics-based simulation -> Signal segmentation -> Embedding optimization -> Recurrence analysis -> Chaos detection -> CNN classification.

---

## Quick Start

### Files You Need
- main_pipeline.py (master script - runs everything)
- ROM_and_segmentation.py
- cao_theorem.py
- average_mutual_information.py
- recurrence_matrix_generation.py
- classification.py
- convolutional_neural_network.py
- No external data files required (generates synthetic data)

### Installation
```bash
pip install numpy scipy matplotlib pandas tqdm scikit-learn torch torchvision
```

### Python Requirements
- Python 3.7 or higher

### How to Run - AUTOMATED (RECOMMENDED)
```bash
python main_pipeline.py
```

This single command will execute all 6 modules sequentially and provide progress updates.

**Total Runtime**: Approximately 45-90 minutes depending on hardware.

### Optional: Skip Specific Modules
```bash
python main_pipeline.py --skip 6          # Skip CNN training
python main_pipeline.py --skip 5 6        # Skip classification and CNN
python main_pipeline.py --skip 1          # Skip simulation (if data exists)
```

### Alternative: Manual Execution (Run Each Module Individually)
```bash
python ROM_and_segmentation.py
python cao_theorem.py
python average_mutual_information.py
python recurrence_matrix_generation.py
python classification.py
python convolutional_neural_network.py
```

---

## Pipeline Architecture

```
+-------------------------------------------------------------+
|  main_pipeline.py (MASTER CONTROLLER)                      |
|  Executes all modules sequentially                         |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Module 1: ROM_and_segmentation.py                          |
|  Simulates thermoacoustic combustion dynamics               |
|  Output: pressure_segments/                                 |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Module 2: cao_theorem.py                                   |
|  Determines optimal embedding dimensions                    |
|  Output: data_with_embed/                                   |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Module 3: average_mutual_information.py                    |
|  Calculates optimal time delays                             |
|  Output: data_with_embed_tau/                               |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Module 4: recurrence_matrix_generation.py                  |
|  Generates recurrence matrices from embedded data           |
|  Output: recurrence_matrices/                               |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Module 5: classification.py                                |
|  Classifies stability regimes using chaos detection         |
|  Output: classified_matrices/                               |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Module 6: convolutional_neural_network.py                  |
|  Trains CNN for automated classification                    |
|  Output: recurrence_matrix_cnn.pth + visualizations         |
+-------------------------------------------------------------+
```

---

## Main Pipeline Features

The main_pipeline.py script provides:

(1) **Automated Execution**: Run all 6 modules with a single command
(2) **Real-time Progress**: See console output from each module as it runs
(3) **Time Tracking**: Monitor execution time for each module
(4) **Error Handling**: Pipeline stops if any module fails
(5) **Module Skipping**: Skip specific modules using --skip flag
(6) **Summary Report**: Final summary with execution times and output locations

### Main Pipeline Usage Examples

**Run complete pipeline:**
```bash
python main_pipeline.py
```

OUTPUT LOCATIONS:
----------------------------------------------------------------------
  Pressure Segments:      /path/to/pressure_segments/
  Data with Embedding:    /path/to/data_with_embed/
  Data with Embed & Tau:  /path/to/data_with_embed_tau/
  Recurrence Matrices:    /path/to/recurrence_matrices/
  Classified Matrices:    /path/to/classified_matrices/
  Trained CNN Model:      /path/to/recurrence_matrix_cnn.pth
  Training Curves:        /path/to/training_curves.png
  Confusion Matrix:       /path/to/confusion_matrix.png
======================================================================
```

---

## Detailed Module Descriptions

### Module 1: ROM_and_segmentation.py

**Purpose**: Simulates thermoacoustic combustion dynamics and segments pressure signals.

**What it does**:
(1) Simulates a thermoacoustic combustor with vortex-flame interactions
(2) Models acoustic waves coupling with unsteady heat release
(3) Segments pressure time series into overlapping windows
(4) Creates reversed copies of each segment for data augmentation

**Physical Model**:
(1) Acoustic Field: 10 modal components (eigenmodes of the combustor)
(2) Vortex Dynamics: Discrete vortices shed from backward-facing step
(3) Heat Release: Impulsive forcing when vortices reach flame location
(4) Flow Conditions: Mean velocity from 7.0 to 10.0 m/s (30 values)

**Key Parameters**:
```python
t_end = 0.2              # Simulation time [seconds]
dt = 5e-5                # Time step [seconds]
window_size = 500        # Points per segment
overlap = 250            # Overlap between windows
U0_range = 7.0 to 10.0   # Flow velocity range [m/s]
```

**Outputs**:
(1) Folder: pressure_segments/
(2) File format: U0_[velocity]_segment_[index].npy
(3) Total files: ~1800 segments (30 velocities x ~30 segments x 2 for reversals)
(4) File content: NumPy array of 500 pressure values

**Manual Execution**:
```bash
python ROM_and_segmentation.py
```

---

### Module 2: cao_theorem.py

**Purpose**: Determines optimal embedding dimension for each pressure segment using Cao's method.

**What it does**:
(1) Applies Cao's algorithm (false nearest neighbors approach)
(2) Calculates E1 and E2 statistics for dimensions 1-20
(3) Finds where E1 saturates (dimension stops changing significantly)
(4) Appends embedding dimension to filename without modifying data

**Theory**:
Cao's method identifies the minimum embedding dimension by measuring how distances between neighbors change as dimension increases. When E1 stops changing (rate < 0.05), we've found the optimal dimension.

**Key Parameters**:
```python
max_dim = 20             # Maximum dimension to test
threshold = 0.05         # Convergence criterion for E1
```

**Inputs**:
(1) Folder: pressure_segments/
(2) Files: U0_[velocity]_segment_[index].npy

**Outputs**:
(1) Folder: data_with_embed/
(2) File format: U0_[velocity]_segment_[index]_embed_[dim].npy
(3) Typical embedding dimensions: 2-8
(4) File content: Original pressure data (unchanged)

**Manual Execution**:
```bash
python cao_theorem.py
```

---

### Module 3: average_mutual_information.py

**Purpose**: Calculates optimal time delay (tau) for phase space reconstruction.

**What it does**:
(1) Computes Average Mutual Information (AMI) between x(t) and x(t+tau)
(2) Finds first local minimum (where information becomes independent)
(3) Applies Savitzky-Golay smoothing for robust minimum detection
(4) Appends tau to filename while preserving embedding dimension

**Theory**:
AMI measures nonlinear correlation. The first minimum indicates when x(t+tau) provides new information independent of x(t), making it optimal for embedding.

**Key Parameters**:
```python
max_lag = 500            # Maximum delay to test
bins = 30                # Histogram bins for probability estimation
normalize = True         # Normalize signal to [0,1]
smooth = True            # Apply Savitzky-Golay filter
```

**Inputs**:
(1) Folder: data_with_embed/
(2) Files: U0_[velocity]_segment_[index]_embed_[dim].npy

**Outputs**:
(1) Folder: data_with_embed_tau/
(2) File format: U0_[velocity]_segment_[index]_embed_[dim]_tau_[value].npy
(3) Typical tau values: 1-50
(4) File content: Original pressure data (unchanged)

**Manual Execution**:
```bash
python average_mutual_information.py
```

---

### Module 4: recurrence_matrix_generation.py

**Purpose**: Generates recurrence matrices from time-delay embedded data.

**What it does**:
(1) Performs time-delay embedding using optimal parameters
(2) Computes pairwise distances in phase space
(3) Creates binary recurrence matrix (1 if distance < threshold)
(4) Resizes all matrices to standard 450x450 size
(5) Generates visualization of first 50 matrices

**Theory**:
A recurrence matrix R(i,j) = 1 if trajectories at times i and j are close in phase space. Patterns reveal:
(1) Diagonal lines: Deterministic dynamics
(2) Vertical/horizontal lines: Laminar states
(3) Isolated points: Stochastic behavior
(4) Large black regions: Intermittency

**Key Parameters**:
```python
matrix_size = (450, 450)     # Standardized output size
threshold = 0.1 * max_dist   # 10% of maximum distance
```

**Inputs**:
(1) Folder: data_with_embed_tau/
(2) Files: U0_[velocity]_segment_[index]_embed_[dim]_tau_[value].npy

**Outputs**:
(1) Folder: recurrence_matrices/
(2) File format: recurrence_matrix_U0_[velocity]_segment_[index].npy
(3) Matrix size: 450x450 binary (0 or 1)
(4) Visualization: first_50_recurrence_matrices.png (5x10 grid)

**Manual Execution**:
```bash
python recurrence_matrix_generation.py
```

---

### Module 5: classification.py

**Purpose**: Classifies combustion stability regimes using geometric analysis and chaos detection.

**What it does**:
(1) Analyzes recurrence matrix patterns geometrically
(2) Applies 0-1 test for chaos to original time series
(3) Classifies into 4 stability types based on dynamical characteristics
(4) Saves only classified matrices (excludes ambiguous cases)
(5) Generates visualizations and detailed statistics

**Classification Scheme**:

**Type 0: Regular Dynamics (K <= 0.2)**
(1) Periodic or quasi-periodic oscillations
(2) Criterion: 0-1 test K-value <= 0.2
(3) Example: Stable combustion with limit cycle

**Type 1: Type I Intermittency - 2nd Quadrant (F_b ~ 1.0, centroid top-left)**
(1) Classic intermittency pattern
(2) Criterion: Large continuous black region (F_b ~ 1.0) with centroid in 2nd quadrant
(3) Example: Bursting behavior with laminar phases

**Type 2: Type I Intermittency - 4th Quadrant (F_b ~ 1.0, centroid bottom-right)**
(1) Reversed intermittency pattern
(2) Criterion: Large continuous black region (F_b ~ 1.0) with centroid in 4th quadrant
(3) Example: Time-reversed bursting dynamics

**Type 3: Chaotic Dynamics (K >= 0.8)**
(1) Fully developed chaos
(2) Criterion: 0-1 test K-value >= 0.8
(3) Example: Turbulent combustion with sensitive dependence

**Unclassified (type_code = -1)**
(1) Ambiguous cases not matching criteria above
(2) These are NOT saved to output folder

**Key Parameters**:
```python
# 0-1 Test
num_c = 100              # Random frequencies to test
c_bounds = (pi/5, 4*pi/5)   # Frequency range
threshold = 0.5          # Chaos boundary

# Type I Detection
F_b_tolerance = 0.1      # |F_b - 1.0| < 0.1
```

**Inputs**:
(1) Folder: recurrence_matrices/
(2) Files: recurrence_matrix_U0_[velocity]_segment_[index].npy
(3) Also reads: pressure_segments/ (for 0-1 test on original signals)

**Outputs**:
(1) Folder: classified_matrices/
(2) File format: recurrence_matrix_U0_[velocity]_segment_[index]_type_[code].npy
(3) CSV: intermittency_analysis_results.csv (detailed metrics)
(4) Visualizations: classified_matrices_visualization_[N].png
(5) Typical retention: ~60-80% of matrices (rest are unclassified)

**Manual Execution**:
```bash
python classification.py
```

---

### Module 6: convolutional_neural_network.py

**Purpose**: Trains a deep convolutional neural network to classify recurrence matrices into stability regimes.

**What it does**:
(1) Loads classified recurrence matrices as training data
(2) Trains a CNN to recognize patterns in 450x450 binary images
(3) Evaluates performance on validation set
(4) Generates training curves and confusion matrix
(5) Saves trained model for future predictions

**Network Architecture**:
```
Input: 1x450x450 (grayscale recurrence matrix)
    |
    v
Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(4x4)  [-> 112x112]
    |
    v
Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)  [-> 56x56]
    |
    v
Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)  [-> 28x28]
    |
    v
Flatten -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.5)
    |
    v
Linear(4) -> Output [4 classes: Types 0, 1, 2, 3]
```

**Training Details**:
(1) Optimizer: Adam (learning rate = 0.001)
(2) Loss Function: CrossEntropyLoss
(3) Batch Size: 16
(4) Epochs: 50
(5) Train/Val Split: 80/20 (fixed random seed = 42)
(6) Hardware: Automatic GPU detection (CUDA if available)

**Key Parameters**:
```python
num_classes = 4          # Types 0, 1, 2, 3
batch_size = 16          # Batch size for training
num_epochs = 50          # Training epochs
learning_rate = 0.001    # Adam optimizer learning rate
dropout = 0.5            # Dropout probability
```

**Inputs**:
(1) Folder: classified_matrices/
(2) Files: recurrence_matrix_U0_[velocity]_segment_[index]_type_[code].npy

**Outputs**:
(1) Model file: recurrence_matrix_cnn.pth (trained weights)
(2) Plot: training_curves.png (loss and accuracy)
(3) Plot: confusion_matrix.png (classification performance)
(4) Console: Per-class accuracy every 50 epochs

**Manual Execution**:
```bash
python convolutional_neural_network.py
```

---

## Output Directory Structure

After running the main pipeline, your directory will contain:

```
project/
|-- main_pipeline.py                      # MASTER SCRIPT
|-- ROM_and_segmentation.py               # Module 1
|-- cao_theorem.py                        # Module 2
|-- average_mutual_information.py         # Module 3
|-- recurrence_matrix_generation.py       # Module 4
|-- classification.py                     # Module 5
|-- convolutional_neural_network.py       # Module 6
|
|-- pressure_segments/                    # Module 1 output
|   |-- U0_7.00_segment_0000.npy
|   |-- U0_7.00_segment_0001.npy
|
|-- data_with_embed/                      # Module 2 output
|   |-- U0_7.00_segment_0000_embed_3.npy
|   |-- U0_7.00_segment_0001_embed_4.npy
|
|-- data_with_embed_tau/                  # Module 3 output
|   |-- U0_7.00_segment_0000_embed_3_tau_5.npy
|   |-- U0_7.00_segment_0001_embed_4_tau_12.npy
|
|-- recurrence_matrices/                  # Module 4 output
|   |-- recurrence_matrix_U0_7.00_segment_0000.npy
|   |-- recurrence_matrix_U0_7.00_segment_0001.npy
|   +-- first_50_recurrence_matrices.png
|
|-- classified_matrices/                  # Module 5 output
|   |-- recurrence_matrix_U0_7.00_segment_0000_type_1.npy
|   |-- recurrence_matrix_U0_7.10_segment_0005_type_3.npy
|   |-- intermittency_analysis_results.csv
|   +-- classified_matrices_visualization_1.png
|
|-- recurrence_matrix_cnn.pth             # Module 6 output
|-- training_curves.png                   # Module 6 output
+-- confusion_matrix.png                  # Module 6 output
```