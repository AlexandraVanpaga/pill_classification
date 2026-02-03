# Pill Classification System

A deep learning project for classifying pharmaceutical pills using CNN models trained on pill images.

---

## Description

The project implements an image classification pipeline for identifying pills by their visual appearance. The system uses fine-tuned EfficientNet as the primary model, with Test Time Augmentation (TTA) to boost inference accuracy. The final model achieves >75% accuracy on the test set.

### Key Features:
- Multi-architecture benchmarking (ResNet, EfficientNet, DenseNet, ViT)
- EfficientNetB4 as the best-performing model (comparable to ViT, but significantly lighter and faster)
- 4-fold augmentation strategy (original + horizontal flip + vertical flip + both flips)
- Test Time Augmentation (TTA) for improved inference accuracy
- Detailed per-class error analysis and confusion matrix
- Experimental results saved to `results/`

---

## Project Structure

```
pill_classification/
├── data/
│   ├── raw/                          # Raw pill images
│   └── processed/                    # Preprocessed and split data
│           ├── train/                # Training set
│           ├── val/                  # Validation set
│           └── test/                 # Test set
├── src/
│   ├── get_raw_data.py               # Raw data download
│   ├── data_preprocessing.py         # Data preprocessing and train/val/test split
│   ├── model_efficientnet.py         # Model architecture definition
│   ├── model_train.py                # Model training
│   └── model_eval.py                 # Model evaluation and error analysis
├── results/                          # Experiment results, plots, confusion matrices
├── models/                           # Saved model checkpoints
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- torchvision
- timm
- Pillow

### Installing Dependencies

```bash
# Windows
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## Usage

### 1. Download Raw Data

```bash
python -m src.get_raw_data
```

### 2. Data Preprocessing

Handles image resizing, normalization, and train/val/test splitting.

```bash
python -m src.data_preprocessing
```

### 3. Model Definition

Initializes the EfficientNetB4 architecture with a custom classification head.

```bash
python -m src.model_efficientnet
```

### 4. Model Training

```bash
python -m src.model_train
```

### 5. Model Evaluation

Runs inference on the test set with TTA, computes per-class metrics, and generates confusion matrices.

```bash
python -m src.model_eval
```

---

## Model Architecture

### Why EfficientNetB4?

| Model | Val Accuracy | Speed | Size |
|-------|-------------|-------|------|
| ResNet | Lower | Fast | Small |
| DenseNet | Lower | Moderate | Moderate |
| ViT | ~83% | Slow | Large |
| **EfficientNetB4** | **~83%** | **Fast** | **Small** |

EfficientNetB4 achieved accuracy comparable to ViT (~83% on validation) while being significantly lighter and faster for both training and inference.

### Augmentation Strategy

Empirically, a 4-fold augmentation approach yielded the best results:

| Augmentation | Description |
|--------------|-------------|
| Original | No transformation |
| Horizontal flip | Mirror along vertical axis |
| Vertical flip | Mirror along horizontal axis |
| Both flips | Horizontal + vertical flip combined |

### Test Time Augmentation (TTA)

During inference, each image is passed through the model 4 times (original + 3 augmented versions). The final prediction is the average of all 4 outputs, which improves classification confidence and accuracy.

```
Image → [Original, H-Flip, V-Flip, HV-Flip]
      → Model × 4
      → Average Scores
      → Final Prediction
```

---

## 📈 Results

### Overall Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | >75% |
| Validation Accuracy | ~83% |
| Best Model | EfficientNetB4 |

### Hardest Classes to Classify

| Class | Reason |
|-------|--------|
| `cataflam_50_mg` | Plain white pill, minimal distinguishing features |
| `normodipine_5_mg` | Visually similar to other white round pills |
| `lactiv_plus` | Nearly identical capsule appearance to `enterol_250_mg` |
| `teva_ambrobene_30_mg` | Low visual contrast between similar pills |
| `enterol_250_mg` | Two capsules nearly identical to `lactiv_plus` |

### Easiest Classes (Zero Errors)

`acc_long_600_mg`, `novo_c_plus`, `kalium_r`, `koleszterin_kontroll`, `lactamed`, `lordestin_5_mg`, `advil_ultra_forte`, `mezym_forte_10_000_egyseg`, `milgamma`, `naprosyn_250_mg`

These pills have strong distinguishing features — unique color, shape, or printed markings — making them easy for the model to identify.

---

## Error Analysis

The most common errors occur on pills that are:
- **Visually identical** — white, round or oval pills with no markings (e.g., `lactiv_plus` vs `enterol_250_mg`)
- **Underrepresented** — classes with fewer training samples lead to weaker learned features

Full confusion matrices and per-class precision/recall curves are saved in `results/`.

---

## Potential Improvements

- **More data for hard classes** — collecting additional samples for the most confused pill classes would directly improve accuracy on those categories
- **Heavier models** — ViT or EfficientNetB7 could capture finer details, though at the cost of training time. An ensemble of ViT + EfficientNet showed promising results
- **Focal Loss** — designed for class imbalance, Focal Loss could help the model focus on hard-to-distinguish pills instead of easy ones
- **Layer-wise feature analysis** — visualizing learned features per layer (e.g., via Grad-CAM) could reveal what the model relies on and guide further feature engineering for difficult classes
- **Confusion-based retraining** — identifying specific class pairs that are frequently confused and augmenting training data specifically for those pairs

---

## Technologies

- **PyTorch** — training framework
- **timm** — pretrained model zoo (EfficientNet, ViT)
- **torchvision** — image augmentation and preprocessing
- **Pillow** — image loading and manipulation
- **scikit-learn** — confusion matrix and classification metrics
- **matplotlib / seaborn** — visualization
EOF
Salida

# Pill Classification System

A deep learning project for classifying pharmaceutical pills using CNN models trained on pill images.

---

## Description

The project implements an image classification pipeline for identifying pills by their visual appearance. The system uses fine-tuned EfficientNet as the primary model, with Test Time Augmentation (TTA) to boost inference accuracy. The final model achieves >75% accuracy on the test set.

### Key Features:
- Multi-architecture benchmarking (ResNet, EfficientNet, DenseNet, ViT)
- EfficientNetB4 as the best-performing model (comparable to ViT, but significantly lighter and faster)
- 4-fold augmentation strategy (original + horizontal flip + vertical flip + both flips)
- Test Time Augmentation (TTA) for improved inference accuracy
- Detailed per-class error analysis and confusion matrix
- Experimental results saved to `results/`

---

## Project Structure

```
pill_classification/
├── data/
│   ├── raw/                          # Raw pill images
│   └── processed/                    # Preprocessed and split data
│           ├── train/                # Training set
│           ├── val/                  # Validation set
│           └── test/                 # Test set
├── src/
│   ├── get_raw_data.py               # Raw data download
│   ├── data_preprocessing.py         # Data preprocessing and train/val/test split
│   ├── model_efficientnet.py         # Model architecture definition
│   ├── model_train.py                # Model training
│   └── model_eval.py                 # Model evaluation and error analysis
├── results/                          # Experiment results, plots, confusion matrices
├── models/                           # Saved model checkpoints
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- torchvision
- timm
- Pillow

### Installing Dependencies

```bash
# Windows
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## Usage

### 1. Download Raw Data

```bash
python -m src.get_raw_data
```

### 2. Data Preprocessing

Handles image resizing, normalization, and train/val/test splitting.

```bash
python -m src.data_preprocessing
```

### 3. Model Definition

Initializes the EfficientNetB4 architecture with a custom classification head.

```bash
python -m src.model_efficientnet
```

### 4. Model Training

```bash
python -m src.model_train
```

### 5. Model Evaluation

Runs inference on the test set with TTA, computes per-class metrics, and generates confusion matrices.

```bash
python -m src.model_eval
```

---

## Model Architecture

### Why EfficientNetB4?

| Model | Val Accuracy | Speed | Size |
|-------|-------------|-------|------|
| ResNet | Lower | Fast | Small |
| DenseNet | Lower | Moderate | Moderate |
| ViT | ~83% | Slow | Large |
| **EfficientNetB4** | **~83%** | **Fast** | **Small** |

EfficientNetB4 achieved accuracy comparable to ViT (~83% on validation) while being significantly lighter and faster for both training and inference.

### Augmentation Strategy

Empirically, a 4-fold augmentation approach yielded the best results:

| Augmentation | Description |
|--------------|-------------|
| Original | No transformation |
| Horizontal flip | Mirror along vertical axis |
| Vertical flip | Mirror along horizontal axis |
| Both flips | Horizontal + vertical flip combined |

### Test Time Augmentation (TTA)

During inference, each image is passed through the model 4 times (original + 3 augmented versions). The final prediction is the average of all 4 outputs, which improves classification confidence and accuracy.

```
Image → [Original, H-Flip, V-Flip, HV-Flip]
      → Model × 4
      → Average Scores
      → Final Prediction
```

---

## 📈 Results

### Overall Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | >75% |
| Validation Accuracy | ~83% |
| Best Model | EfficientNetB4 |

### Hardest Classes to Classify

| Class | Reason |
|-------|--------|
| `cataflam_50_mg` | Plain white pill, minimal distinguishing features |
| `normodipine_5_mg` | Visually similar to other white round pills |
| `lactiv_plus` | Nearly identical capsule appearance to `enterol_250_mg` |
| `teva_ambrobene_30_mg` | Low visual contrast between similar pills |
| `enterol_250_mg` | Two capsules nearly identical to `lactiv_plus` |

### Easiest Classes (Zero Errors)

`acc_long_600_mg`, `novo_c_plus`, `kalium_r`, `koleszterin_kontroll`, `lactamed`, `lordestin_5_mg`, `advil_ultra_forte`, `mezym_forte_10_000_egyseg`, `milgamma`, `naprosyn_250_mg`

These pills have strong distinguishing features — unique color, shape, or printed markings — making them easy for the model to identify.

---

## Error Analysis

The most common errors occur on pills that are:
- **Visually identical** — white, round or oval pills with no markings (e.g., `lactiv_plus` vs `enterol_250_mg`)
- **Underrepresented** — classes with fewer training samples lead to weaker learned features

Full confusion matrices and per-class precision/recall curves are saved in `results/`.

---

## Potential Improvements

- **More data for hard classes** — collecting additional samples for the most confused pill classes would directly improve accuracy on those categories
- **Heavier models** — ViT or EfficientNetB7 could capture finer details, though at the cost of training time. An ensemble of ViT + EfficientNet showed promising results
- **Focal Loss** — designed for class imbalance, Focal Loss could help the model focus on hard-to-distinguish pills instead of easy ones
- **Layer-wise feature analysis** — visualizing learned features per layer (e.g., via Grad-CAM) could reveal what the model relies on and guide further feature engineering for difficult classes
- **Confusion-based retraining** — identifying specific class pairs that are frequently confused and augmenting training data specifically for those pairs

---

## Technologies

- **PyTorch** — training framework
- **timm** — pretrained model zoo (EfficientNet, ViT)
- **torchvision** — image augmentation and preprocessing
- **Pillow** — image loading and manipulation
- **scikit-learn** — confusion matrix and classification metrics
- **matplotlib / seaborn** — visualization
