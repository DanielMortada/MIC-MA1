#!/usr/bin/env python
"""
Training script for ResNet-18 baseline model for musical instrument classification.
Usage: python scripts/train_baseline.py

This script trains a ResNet-18 model using the configuration specified in
config/baseline_resnet18.yaml.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# Add project root to path to ensure imports work correctly
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

# Import project modules
from src.data.preprocessing import get_preprocessing_transforms
from src.data.dataloader import load_datasets
from src.models.baseline import get_resnet18_model
from src.training.trainer import train_model, evaluate_model
from src.training.metrics import compute_metrics, get_confusion_matrix
from src.visualization.plotting import plot_training_history, plot_confusion_matrix
from src.models.model_utils import save_model

def main():
    # Load configuration
    config_path = os.path.join(project_root, "config", "baseline_resnet18.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Print configuration details
    print("Training ResNet-18 baseline with configuration:")
    print(f"- Epochs: {config['training']['num_epochs']}")
    print(f"- Batch size: {config['training']['batch_size']}")
    print(f"- Learning rate: {config['training']['optimizer']['learning_rate']}")
    print(f"- Optimizer: {config['training']['optimizer']['name']}")
    print(f"- Scheduler: {config['training']['scheduler']['name']}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get preprocessing transforms
    transforms = get_preprocessing_transforms(img_size=config['data']['img_size'])
    
    # Load datasets
    data_dir = os.path.join(project_root, config['data']['data_dir'])
    data = load_datasets(
        data_dir, 
        transforms,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    dataloaders = data['dataloaders']
    class_names = list(data['class_mappings']['idx_to_class'].values())
    num_classes = data['num_classes']
    
    print(f"Dataset loaded: {num_classes} classes")
    print(f"Training samples: {len(data['datasets']['train'])}")
    print(f"Validation samples: {len(data['datasets']['val'])}")
    print(f"Test samples: {len(data['datasets']['test'])}")
    
    # Create model
    model = get_resnet18_model(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        feature_extracting=config['model']['feature_extracting']
    )
    model = model.to(device)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer
    optimizer_config = config['training']['optimizer']
    optimizer_name = optimizer_config['name'].lower()
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            momentum=0.9,
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Set up scheduler
    scheduler_config = config['training']['scheduler']
    scheduler_name = scheduler_config['name'].lower()
    
    if scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config['factor'],
            patience=scheduler_config['patience'],
            min_lr=scheduler_config['min_lr']
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=7,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # Train model
    print("\nStarting training...")
    model, history, training_stats = train_model(
        model,
        dataloaders={
            'train': dataloaders['train'],
            'val': dataloaders['val']
        },
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        num_epochs=config['training']['num_epochs'],
        verbose=True
    )
    
    # Print training summary
    print("\nTraining complete!")
    print(f"Best validation accuracy: {training_stats['best_val_acc']:.4f} at epoch {training_stats['best_epoch']}")
    print(f"Training time: {training_stats['training_time']}")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_preds, test_labels = evaluate_model(
        model,
        dataloaders['test'],
        device,
        verbose=True
    )
    
    # Compute metrics
    metrics = compute_metrics(test_labels, test_preds, class_names)
    
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-average F1 Score: {metrics['macro_avg']['f1']:.4f}")
    
    # Plot confusion matrix
    cm = get_confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, class_names, figsize=(12, 10), fontsize=8)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(project_root, "experiments", "baseline_resnet18")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"resnet18_baseline_{timestamp}.pt")
    save_model(
        model,
        config,
        save_path,
        metrics={
            'accuracy': test_accuracy,
            'best_val_acc': training_stats['best_val_acc'],
            'best_epoch': training_stats['best_epoch']
        }
    )
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()
