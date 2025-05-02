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
- Advanced model architectures with attention mechanisms and residual connections

## Project Structure

```bash
proj-h419-MIC/
├── config/                  # Configuration files for experiments
│   ├── baseline_resnet18.yaml
│   ├── custom_cnn_base.yaml
│   ├── custom_cnn_deeper.yaml
│   ├── custom_cnn_regularized.yaml
│   ├── custom_cnn_wider.yaml
│   ├── custom_model_v1.yaml
│   ├── flexible_framework.yaml
│   ├── optimized_custom_cnn.yaml
│   ├── optimized_deeper_cnn.yaml
│   └── resnet18_baseline.yaml
├── data/                    # Dataset directory
│   ├── processed/           # Processed dataset
│   └── raw/                 # Raw dataset with 30 Musical Instruments
├── diagrams/                # Project architecture and pipeline diagrams
│   └── TrainingPipeline.mermaid
├── docs/                    # Documentation and learning resources
│   ├── cnn_optimization_strategies.md
│   ├── crash_course.md
│   ├── diff_scheduler_optimizer.md
│   ├── model_architecture_comparison.md
│   ├── pipeline.md
│   └── training_and_eval_basics.md
├── notebooks/               # Jupyter notebooks for exploration and visualization
│   ├── 1-Dataset Acquisition and Exploration.ipynb
│   ├── 2_Baseline_ResNet18.ipynb
│   ├── 3_Model_From_Scratch_v1_fixed.ipynb
│   ├── 4_Flexible_Model_Comparison.ipynb
│   ├── 5_Custom_Model_Optimization.ipynb
│   ├── 6_Deeper_CNN_Optimisation.ipynb
│   └── res/                 # Notebook resources and outputs
├── report/                  # Project report and presentation
├── scripts/                 # Training and evaluation scripts
│   ├── colab_integration.py
│   ├── evaluate_model.py
│   ├── train_baseline.py
│   ├── train_custom_cnn.py
│   ├── train_flexible.py
│   └── train_parallel.py
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   │   ├── augmentation.py
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/             # Model architecture modules
│   │   ├── attention.py
│   │   ├── baseline.py
│   │   ├── custom_cnn.py
│   │   ├── dataparallel_utils.py
│   │   ├── enhanced_cnn.py
│   │   ├── enhanced_flexible_cnn.py
│   │   ├── flexible_cnn.py
│   │   └── model_utils.py
│   ├── training/           # Training utilities
│   │   ├── enhanced_trainer.py
│   │   ├── metrics.py
│   │   ├── scheduler.py
│   │   └── trainer.py
│   └── visualization/      # Visualization utilities
│       └── plotting.py
└── tests/                  # Test results and experiment outputs
    ├── base-cnn-20250430.085509/
    ├── deeper-cnn-20250430.093116/
    ├── optimized_deeper_cnn/
    ├── regularized-cnn-20250430.111203/
    ├── resnet_18.20250430.114435/
    └── wider-cnn-20250430.103336/
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
python scripts/train_baseline.py --config config/baseline_resnet18.yaml
```

#### Training a custom CNN model

```bash
python scripts/train_custom_cnn.py --config config/custom_cnn_base.yaml
```

#### Training with multiple model variations

```bash
python scripts/train_flexible.py --config config/flexible_framework.yaml
```

#### Evaluating a trained model

```bash
python scripts/evaluate_model.py --model_path tests/optimized_deeper_cnn/best_model.pth
```

#### Using Google Colab for GPU acceleration

You can run the training on Google Colab to leverage GPU acceleration:

1. Clone this repository directly in Colab
2. Use the setup and GPU detection utilities:

```python
from scripts.colab_integration import setup_colab_environment, check_gpu

# Set up dependencies
setup_colab_environment()

# Check for GPU and get the appropriate device
device = check_gpu()

# Your device will automatically be set to use GPU if available
```

## Model Architectures

### Baseline Model (ResNet18)
We use a pre-trained ResNet18 model with a modified classifier head to establish a strong baseline.

### Custom CNN
Our custom CNN architecture features multiple convolutional blocks with batch normalization, dropout, and a global pooling strategy.

### Enhanced Models
We've implemented several enhanced architectures:

1. **Deeper CNN**: More convolutional layers for better feature extraction
2. **Wider CNN**: More filters per layer for capturing diverse features
3. **Regularized CNN**: Enhanced dropout and augmentation for better generalization
4. **Attention-Enhanced CNN**: Incorporating attention mechanisms to focus on important features
5. **Residual CNN**: Including residual connections for better gradient flow

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
- **custom_cnn_base.yaml**: Parameters for training our basic custom CNN architecture
- **custom_cnn_deeper.yaml**: Configuration for deeper network architecture
- **custom_cnn_wider.yaml**: Configuration for wider network architecture
- **custom_cnn_regularized.yaml**: Enhanced parameters with regularization techniques
- **optimized_deeper_cnn.yaml**: Optimized deeper CNN model with advanced features

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

## Model Optimization

The project includes a systematic approach to optimize neural network performance:

### Optimization Techniques

- **Enhanced Data Augmentation**: Random erasing, stronger color jittering, and spatial transformations
- **Advanced Learning Rate Scheduling**: OneCycle policy for faster convergence
- **Regularization Techniques**: Gradient clipping, adaptive dropout, and early stopping
- **Improved Weight Decay**: AdamW optimizer with properly configured weight decay
- **Architecture Enhancements**: Attention mechanisms and residual connections
- **Hyperparameter Tuning**: Systematic exploration of key parameters

### Optimization Workflow

1. Start with a baseline model (either ResNet18 or custom CNN)
2. Apply the optimized configuration from `optimized_custom_cnn.yaml`
3. Run training with enhanced regularization techniques
4. Evaluate performance improvements through comprehensive metrics
5. Analyze learning dynamics via visualization tools

For a detailed guide on CNN optimization, see `docs/cnn_optimization_strategies.md`.

## Jupyter Notebooks

The `notebooks/` directory contains interactive explorations of the project:

1. **1-Dataset Acquisition and Exploration.ipynb**: Initial dataset inspection, class distribution analysis, and sample visualization
2. **2_Baseline_ResNet18.ipynb**: Implementation of the ResNet18 transfer learning approach with performance evaluation
3. **3_Model_From_Scratch_v1_fixed.ipynb**: Development and training of custom CNN architecture from first principles
4. **4_Flexible_Model_Comparison.ipynb**: Demonstration of the flexible training framework using various model architectures
5. **5_Custom_Model_Optimization.ipynb**: Advanced optimization techniques applied to the custom CNN model
6. **6_Deeper_CNN_Optimisation.ipynb**: Implementation of deeper architectures with attention mechanisms and residual connections

## Contributors

- [Your Name]
- [Team Member 1]
- [Team Member 2]

## Acknowledgments

- The dataset used in this project
- PyTorch library and community
- PROJ-H-419 course staff and colleagues
