# CAD Code Generation with Deep Learning

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A state-of-the-art system for generating CADQuery code from images using Vision Transformers and CodeT5, with automatic GPU/CPU adaptation.

## Prerequisites

- Python 3.9+
- Anaconda/Miniconda (recommended)
- NVIDIA GPU (optional but recommended for training)

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cad-codegen.git
cd cad-codegen
```
### 2. Setup Environment
```bash
conda create -n cadcodegen python=3.9
conda activate cadcodegen
pip install -r requirements.txt
```
### 3. Download Dataset (Automatic)
The code will automatically download and cache the dataset from HuggingFace on first run.

Training the Model
Basic Training (Auto-detects GPU/CPU)
```bash
python scripts/train.py
```
Advanced Options
```bash
# Force CPU mode with reduced parameters
USE_GPU=False python scripts/train.py

# Custom batch size
BATCH_SIZE=8 python scripts/train.py

# Debug mode (runs quick validation)
DEBUG=True python scripts/train.py
```
Evaluation
```bash
python scripts/evaluate.py \
    --checkpoint ./models/best_model.pth \
    --samples 100
```
Custom Configuration
Modify the YAML configs in configs/:

yaml
```bash
# configs/train.yaml
training:
  learning_rate: 5e-5
  batch_size: 16
  epochs: 10

data:
  max_length: 512
  image_size: 384
```
Project Structure
```text
cad-codegen/
├── configs/             # Hydra configuration files
├── data/                # Dataset loading utilities
├── models/              # Model architectures
├── training/            # Training loops
├── evaluation/          # Metrics and validation
├── scripts/             # Main executable scripts
├── utils/               # Helper functions
├── .gitignore
├── requirements.txt
└── README.md
```
Hardware Adaptation
The system automatically adjusts:

Model size (ViT-base → ViT-small on CPU)

Batch size (16 → 4 on CPU)

Image resolution (384 → 224 on CPU)

Mixed precision (disabled on CPU)

License
MIT License - See LICENSE for details.

Contributing
Pull requests welcome! Please:

Fork the repository

Create your feature branch

Commit your changes

Push to the branch

Open a pull request


