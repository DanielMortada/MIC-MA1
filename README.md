# Musical Instrument Classification (MIC)

A computer vision project for multi-class classification of musical instruments using deep learning techniques.

## Project Overview

This research project implements and evaluates various deep learning models for classifying images of 30 different musical instruments. The work combines both state-of-the-art transfer learning approaches and custom CNN architectures with advanced optimization techniques.

### Research Objectives

1. Establish a strong baseline using transfer learning (ResNet-18)
2. Develop custom CNN architectures optimized for musical instrument classification
3. Analyze model performance through detailed comparative evaluation
4. Implement and validate advanced architectural enhancements, including attention mechanisms and residual connections
5. Apply systematic optimization strategies to close the performance gap between custom models and transfer learning approaches

## Repository Structure

```
proj-h419-MIC/
├── config/                  # Experiment configuration files
├── data/                    # Dataset directory
│   ├── processed/           # Preprocessed data
│   └── raw/                 # Raw dataset (30 Musical Instruments)
├── notebooks/               # Research notebooks with implementations and analysis
│   ├── 1-Dataset_Exploration.ipynb
│   ├── 2_Baseline_ResNet18.ipynb
│   ├── 3_Model_From_Scratch.ipynb
│   ├── 4_Flexible_Model_Comparison.ipynb
│   ├── 5_Custom_Model_Optimization.ipynb
│   ├── 6_Deeper_CNN_Optimisation.ipynb
│   └── res/                 # Results and analysis artifacts
├── src/                     # Source code
│   ├── data/                # Data processing modules
│   ├── models/              # Model architectures
│   ├── training/            # Training utilities
│   └── visualization/       # Visualization tools
├── scripts/                 # Training and evaluation scripts
└── tests/                   # Experiment outputs and results
```

## Requirements and Setup

### Dependencies

- Python 3.8+
- PyTorch 1.13+
- torchvision 0.13+
- scikit-learn
- matplotlib
- seaborn
- PyYAML
- tqdm

### Installation

```
pip install -r requirements.txt
```

### Dataset Preparation

The project uses the 30 Musical Instruments dataset, which should be organized with the following structure:

```
data/raw/30_Musical_Instruments/
├── train/  # Training data with class subfolders
├── valid/  # Validation data with class subfolders
└── test/   # Test data with class subfolders
```

## Experiments and Model Architecture

### Running Experiments

The project includes several training scripts for different model architectures:

```bash
# Train ResNet-18 baseline
python scripts/train_baseline.py --config config/baseline_resnet18.yaml

# Train custom CNN models
python scripts/train_custom_cnn.py --config config/custom_cnn_base.yaml

# Train multiple architectures for comparison
python scripts/train_flexible.py --config config/flexible_framework.yaml

# Evaluate a trained model
python scripts/evaluate_model.py --model_path tests/optimized_deeper_cnn/best_model.pth
```

### Model Architectures

Our research systematically evaluates the following model architectures:

1. **ResNet-18 (Baseline)**: Transfer learning approach with a pre-trained model and fine-tuning

2. **Custom CNN Variants**:
   - **Base CNN**: Standard convolutional architecture with batch normalization
   - **Deeper CNN**: Extended architecture with additional convolutional layers
   - **Wider CNN**: Architecture with increased filter counts in each layer
   - **Regularized CNN**: Base architecture with enhanced regularization techniques

3. **Optimized Deeper CNN**: Our most advanced architecture featuring:
   - Selective attention mechanisms
   - Residual connections
   - Graduated dropout strategy
   - Class-specific augmentation
   - Mixed precision training

## Results Summary

Our research provides several key findings:

1. **Model Performance Comparison**:
   - ResNet-18 (Transfer Learning): 99.33% test accuracy
   - Original Deeper CNN: 86.67% test accuracy
   - Optimized Deeper CNN: 93.33% test accuracy

2. **Key Architectural Improvements**:
   - Selective attention mechanisms improved feature focus
   - Residual connections enhanced gradient flow
   - Graduated dropout strategy provided better regularization

3. **Training Optimizations**:
   - AdamW optimizer with weight decay
   - OneCycleLR learning rate scheduling
   - Mixed precision training reduced computation time

4. **Class-Specific Performance**:
   - Most challenging instruments: Alphorn, Flute, Clarinet, Didgeridoo
   - Best-performing instruments: Accordion, Banjo, Cello, Trumpet

## Research Methodology

Our approach follows a systematic optimization strategy:

1. **Baseline Establishment**: We begin with transfer learning using ResNet-18 to establish a strong performance baseline

2. **Architecture Exploration**: We systematically evaluate multiple custom CNN architectures with varying depth, width, and regularization strategies

3. **Strategic Optimization**: Based on performance analysis, we identify the Deeper CNN as the most promising architecture and apply targeted optimization techniques:
   - Architecture modifications through selective attention and residual connections
   - Learning dynamics improvements via advanced optimizers and schedulers
   - Data utilization enhancement with class-specific augmentation
   - Computation efficiency with mixed precision training

4. **Comprehensive Evaluation**: We provide detailed performance analysis including:
   - Confusion matrices to identify class-specific performance
   - Learning curve analysis to understand optimization effectiveness
   - F1-score analysis to evaluate balanced performance
   - Computational efficiency metrics

For detailed research findings, please refer to the notebooks in the `notebooks/` directory, particularly `6_Deeper_CNN_Optimisation.ipynb` for our most advanced model.

## Conclusion

This project demonstrates that carefully optimized custom CNN architectures can approach the performance of transfer learning models while maintaining architectural simplicity and interpretability. Our optimized Deeper CNN achieves 93.33% test accuracy, significantly closing the gap with the ResNet-18 baseline (99.33%).
