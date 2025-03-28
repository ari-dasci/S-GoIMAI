# Deep Learning Methodology for Wood Species Identification using High-Resolution Macroscopic Images and Patch-Voting

A deep learning framework for timber classification using macroscopic images with patch-based ensemble voting.

## Key Features
- 🔬 High-resolution macroscopic image processing
- 🗳️ Patch-based majority voting system

## Installation
```bash
# Clone repository
git clone https://github.com/ari-dasci/S-GoIMAI.git
cd S-GoIMAI
```

## Dataset
The GOIMAI Phase-I dataset of high-resolution macroscopic wood images is available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11092208.svg)](https://doi.org/10.5281/zenodo.11092208)

Download the dataset from:  
https://zenodo.org/records/11092208

This expanded version contains:
- 8.7 GB of high-resolution macroscopic images
- 37 wood species with multiple samples each
- 24x magnification captures using specialized equipment

After downloading, unzip the dataset and use the directory structure as:
```
raw_dataset/
├── species_1/
│   ├── image1.jpg
│   └── image2.jpg
└── species_2/
    ├── image1.jpg
    └── image2.jpg
```

## Usage

### 1. Data Augmentation
Generate augmented dataset from original images:
```bash
python scripts/create_dataset.py \
  -i path/to/original_dataset \
  -o path/to/augmented_output \
  -n 10 \                 # 10 augmented versions per image
  -t 8 \                  # Use 8 CPU threads
  -v                       # Verbose output
```

### 2. Model Training
Train a classification model with k-fold cross-validation:
```bash
python scripts/train.py \
  --input path/to/augmented_dataset \
  --original path/to/original_dataset \
  --model b0 \            # EfficientNetV2-B0 architecture
  --size 300 300 \        # Input image dimensions
  --output path/to/models \
  --epochs 100 \
  --batch 32 \
  --k_fold 5 \            # 5-fold cross-validation
  --loss sparse_categorical_crossentropy
```

#### Key Arguments:
| Argument | Description | Example |
|----------|-------------|---------|
| `--input` | Path to augmented dataset | `path/augmented_data` |
| `--original` | Path to original unaugmented dataset | `path/original_data` |
| `--model` | Model architecture (`b0`,`b1`,`b2`,`b3`,`ir2`) | `b2` |
| `--size` | Input dimensions (width height) | `224 224` |
| `--k_fold` | Cross-validation folds | `5` |
| `--output` | Model output directory | `path/models` |

### 3. Output Structure
After training, models are saved with metadata:
```
models/
└── 2024-06-25_14:30:00-b0-sparse_categorical_crossentropy-100epochs/
    ├── fold_0-b0.keras
    ├── fold_0-b0-history.csv
    └── metadata.json
```

## Features
- 🌀 Automated data augmentation pipeline
- 🔢 Stratified k-fold cross-validation
- 📈 Cosine learning rate decay
- 🧠 Support for EfficientNetV2 and InceptionResNetV2
- 📊 Patch-based evaluation with majority voting

## Requirements
- Python 3.8+
- TensorFlow 2.10+
- Pillow
- scikit-learn
- pandas
