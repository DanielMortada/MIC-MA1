#!/usr/bin/env python
"""
Flexible training script for musical instrument classification.
Usage: python scripts/train_flexible.py [--config CONFIG_PATH]

This script provides a flexible framework for training different model architectures
using the configuration specified in config/flexible_framework.yaml or a custom config.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import argparse
from datetime import datetime
from pathlib import Path

# Import project modules
from src.data.preprocessing import get_preprocessing_transforms
from src.data.augmentation import AdvancedAugmentation
from src.data.dataloader import load_datasets
from src.models.baseline import get_resnet18_model, unfreeze_layers
from src.models.custom_cnn import create_custom_cnn
from src.training.trainer import train_model, evaluate_model
from src.training.metrics import compute_metrics, get_confusion_matrix
from src.visualization.plotting import plot_training_history, plot_confusion_matrix, plot_sample_predictions
from src.models.model_utils import save_model

# Add project root to path to ensure imports work correctly
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Flexible Training Framework")
    parser.add_argument(
        "--config", 
        type=str,
        default="config/flexible_framework.yaml",
        help="Path to configuration file"
    )
    return parser.parse_args()

def create_model(config, device):
    """Create model based on configuration"""
    model_config = config['model']
    architecture = model_config.get('architecture', 'resnet18').lower()
    num_classes = model_config.get('num_classes', 30)
    
    if architecture == 'resnet18':
        pretrained = model_config.get('pretrained', True)
        feature_extracting = model_config.get('feature_extracting', True)
        
        model = get_resnet18_model(
            num_classes=num_classes,
            pretrained=pretrained,
            feature_extracting=feature_extracting
        )
        
        # Unfreeze specific layers if specified
        if not feature_extracting and 'unfreeze_layers' in model_config:
            model, _ = unfreeze_layers(model, model_config['unfreeze_layers'])
            
    elif architecture == 'custom_cnn':
        model = create_custom_cnn(
            num_classes=num_classes,
            input_channels=model_config.get('input_channels', 3)
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    return model.to(device)

def create_optimizer(config, model_parameters):
    """Create optimizer based on configuration"""
    optimizer_config = config['training']['optimizer']
    optimizer_name = optimizer_config.get('name', 'adam').lower()
    lr = optimizer_config.get('learning_rate', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(
                optimizer_config.get('beta1', 0.9),
                optimizer_config.get('beta2', 0.999)
            )
        )
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(
                optimizer_config.get('beta1', 0.9),
                optimizer_config.get('beta2', 0.999)
            )
        )
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model_parameters,
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=optimizer_config.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def create_scheduler(config, optimizer, steps_per_epoch=None):
    """Create learning rate scheduler based on configuration"""
    scheduler_config = config['training']['scheduler']
    scheduler_name = scheduler_config.get('name', '').lower()
    
    if not scheduler_name:
        return None
        
    if scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 7),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('t_max', config['training']['num_epochs']),
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 5),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
    elif scheduler_name == 'onecycle' and steps_per_epoch:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_config.get('max_lr', 0.01),
            steps_per_epoch=steps_per_epoch,
            epochs=config['training']['num_epochs'],
            pct_start=scheduler_config.get('pct_start', 0.3)
        )
    else:
        print(f"Warning: Scheduler {scheduler_name} not configured correctly, using no scheduler")
        return None

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config_path = os.path.join(project_root, args.config)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Extract experiment configuration
    architecture = config['model']['architecture']
    experiment_name = f"{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Print configuration details
    print(f"Starting flexible training framework with architecture: {architecture}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Optimizer: {config['training']['optimizer']['name']}")
    if 'scheduler' in config['training'] and config['training']['scheduler'].get('name'):
        print(f"Scheduler: {config['training']['scheduler']['name']}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get preprocessing transforms
    if config['augmentation'].get('augmentation_strength', None):
        print(f"Using advanced augmentation with strength: {config['augmentation']['augmentation_strength']}")
        transforms = AdvancedAugmentation.get_advanced_transforms(
            img_size=config['data']['img_size'],
            augmentation_strength=config['augmentation']['augmentation_strength']
        )
    else:
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
    model = create_model(config, device)
    print(f"Model created: {config['model']['architecture']}")
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer
    optimizer = create_optimizer(config, model.parameters())
    print(f"Optimizer created: {config['training']['optimizer']['name']}")
    
    # Set up scheduler
    scheduler = create_scheduler(
        config, 
        optimizer, 
        steps_per_epoch=len(dataloaders['train'])
    )
    if scheduler:
        print(f"Scheduler created: {config['training']['scheduler']['name']}")
    
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
    
    # Sample predictions
    plot_sample_predictions(
        model,
        dataloaders['test'],
        class_names,
        device,
        num_samples=8
    )
    
    # Save experiment results
    if config['experiment'].get('save_model', True):
        save_dir = os.path.join(
            project_root, 
            config['experiment'].get('save_dir', 'experiments/flexible_framework')
        )
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{experiment_name}.pt")
        save_model(
            model,
            config,
            save_path,
            metrics={
                'accuracy': test_accuracy,
                'f1_macro': metrics['macro_avg']['f1'],
                'best_val_acc': training_stats['best_val_acc'],
                'best_epoch': training_stats['best_epoch']
            },
            epoch=training_stats['best_epoch']
        )
        
        print(f"Model saved to {save_path}")
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()
