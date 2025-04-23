# Musical Instrument Classification (MIC)

A deep learning project for classifying images of musical instruments using convolutional neural networks.

## Project Overview

This project implements and evaluates deep learning models for classifying images of 30 different musical instruments. It serves as part of the PROJ-H-419 course requirements.

### Key Features

- Multi-class image classification (30 musical instrument classes)
- Implementation of both transfer learning (ResNet18) and custom CNN architectures
- Comprehensive data preprocessing and augmentation pipeline
- Modular, reusable codebase structure
- Experiment tracking and reproducible configurations
- Performance evaluation with detailed metrics and visualizations

## Project Structure

```bash
proj-h419-MIC/
├── config/                  # Configuration files for experiments
│   ├── baseline_resnet18.yaml
│   ├── custom_model_v1.yaml
│   └── flexible_framework.yaml
├── data/                    # Dataset directory
│   ├── processed/           # Processed dataset
│   └── raw/                 # Raw dataset with 30 Musical Instruments
├── diagrams/               # Project architecture and pipeline diagrams
│   └── TrainingPipeline.mermaid
├── docs/                    # Documentation and learning resources
│   ├── crash_course.md
│   ├── diff_scheduler_optimizer.md
│   ├── model_architecture_comparison.md
│   ├── pipeline.md
│   └── training_and_eval_basics.md
├── experiments/             # Experiment results and saved models
├── notebooks/              # Jupyter notebooks for exploration and visualization
│   ├── 1-Dataset Acquisition and Exploration.ipynb
│   ├── 2-Baseline_ResNet18.ipynb
│   ├── 3-Model_From_Scratch_v1.ipynb
│   ├── 4-Flexible_Model_Framework.ipynb
│   ├── training_summary.md
│   └── res/                # Notebook resources and outputs
├── report/                  # Project report and presentation
├── scripts/                # Training and evaluation scripts
│   ├── colab_integration.py
│   ├── evaluate_model.py
│   ├── train_baseline.py
│   ├── train_custom_cnn.py
│   └── train_flexible.py
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   ├── models/             # Model architecture modules
│   ├── training/           # Training utilities
│   └── visualization/      # Visualization utilities
└── tests/                  # Unit tests
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.13+
- CUDA (optional, but recommended for GPU acceleration)

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the dataset:
   - Place the raw 30 Musical Instruments dataset in `data/raw/`
   - The data should be organized with train/valid/test splits

### Running Experiments

#### Training the ResNet18 baseline model

```bash
python scripts/train_baseline.py
```

#### Training a custom CNN model

```bash
python scripts/train_custom_cnn.py
```

#### Using Google Colab for GPU acceleration

You can run the training on Google Colab to leverage GPU acceleration:

1. Upload the project zip file to Colab
2. Mount your Google Drive if needed for data access
3. Use the GPU detection utility for automatic acceleration:

```python
from scripts.colab_integration import check_gpu

# Check for GPU and get the appropriate device
device = check_gpu()

# Your device will automatically be set to use GPU if available
```

## Results

Results from our model experiments will be published upon completion of the project.

## Configuration System

The project uses a YAML-based configuration system to manage experiment parameters. This approach offers:

- **Reproducibility**: Each experiment can be precisely recreated using the same config file
- **Flexibility**: Easy modification of hyperparameters without changing code
- **Documentation**: Configurations serve as a record of experiment parameters
- **Standardization**: Common structure for all experiment types

### Configuration Files

The `config/` directory contains several configuration templates:

- **baseline_resnet18.yaml**: Parameters for training the ResNet18 transfer learning model
- **custom_model_v1.yaml**: Parameters for training our custom CNN architecture
- **flexible_framework.yaml**: Dynamic configuration supporting multiple architectures and training strategies

### Using Configurations

To use a configuration file with the flexible training framework:

```bash
python scripts/train_flexible.py --config config/flexible_framework.yaml
```

### Key Configuration Parameters

Configuration files include sections for:

- **Model**: Architecture type, pretrained options, layer freezing/unfreezing
- **Training**: Epochs, batch size, loss function
- **Optimizer**: Type (SGD, Adam, AdamW), learning rate, weight decay
- **Scheduler**: Learning rate scheduling strategy
- **Data**: Augmentation strategies, preprocessing steps
- **Evaluation**: Metrics to track during training

## Jupyter Notebooks

The `notebooks/` directory contains interactive explorations of the project:

1. **1-Dataset Acquisition and Exploration.ipynb**: Initial dataset inspection, class distribution analysis, and sample visualization
2. **2-Baseline_ResNet18.ipynb**: Implementation of the ResNet18 transfer learning approach with performance evaluation
3. **3-Model_From_Scratch_v1.ipynb**: Development and training of custom CNN architecture from first principles
4. **4-Flexible_Model_Framework.ipynb**: Demonstration of the flexible training framework using various model architectures

## Acknowledgments

- The dataset used in this project
- PyTorch library and community
- PROJ-H-419 course staff and colleagues
