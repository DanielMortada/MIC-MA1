#!/usr/bin/env python
"""
Model evaluation script for musical instrument classification.
Usage: python scripts/evaluate_model.py --model_path PATH_TO_MODEL

This script evaluates a trained model on the test dataset and 
generates detailed performance metrics and visualizations.
"""
import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
# Import project modules
from src.data.preprocessing import get_preprocessing_transforms
from src.data.augmentation import AdvancedAugmentation
from src.data.dataloader import load_datasets
from src.training.metrics import compute_metrics, get_confusion_matrix
from src.visualization.plotting import (
    plot_confusion_matrix, 
    plot_sample_predictions
)
from src.models.model_utils import load_saved_model

# Add project root to path to ensure imports work correctly
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument(
        "--model_path", 
        type=str,
        required=True,
        help="Path to saved model (.pt file)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Path to data directory"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed class-wise metrics"
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the saved model
    print(f"Loading model from {args.model_path}")
    model, config, saved_metrics, epoch = load_saved_model(args.model_path, device)
    
    # Print model information
    print("Model information:")
    if config and 'model' in config:
        for key, value in config['model'].items():
            print(f"- {key}: {value}")
    
    if saved_metrics:
        print("\nSaved metrics:")
        for key, value in saved_metrics.items():
            if isinstance(value, (int, float)):
                print(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
    
    # Set up data preprocessing and loading
    data_dir = os.path.join(project_root, args.data_dir)
    
    # Determine transforms from config if available
    if config and 'data' in config:
        img_size = config['data'].get('img_size', 224)
        
        # Use advanced augmentation if specified
        if (config.get('augmentation', {}).get('augmentation_strength')):
            transforms = AdvancedAugmentation.get_advanced_transforms(
                img_size=img_size,
                augmentation_strength=config['augmentation']['augmentation_strength']
            )
        else:
            transforms = get_preprocessing_transforms(img_size=img_size)
    else:
        # Default transforms
        transforms = get_preprocessing_transforms(img_size=224)
    
    # Load datasets
    batch_size = args.batch_size
    data = load_datasets(
        data_dir, 
        transforms,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    dataloaders = data['dataloaders']
    class_names = list(data['class_mappings']['idx_to_class'].values())
    
    
    # Set the model to evaluation mode
    model.eval()
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Compute detailed metrics
    metrics = compute_metrics(all_labels, all_preds, class_names)
    
    print(f"Macro F1-score: {metrics['macro_avg']['f1']:.4f}")
    print(f"Weighted F1-score: {metrics['weighted_avg']['f1']:.4f}")
    
    if args.detailed:
        print("\nClass-wise metrics:")
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 60)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<20} {metrics['class_metrics']['precision'][i]:.4f}     {metrics['class_metrics']['recall'][i]:.4f}     {metrics['class_metrics']['f1'][i]:.4f}     {metrics['class_metrics']['support'][i]}")
    
    # Plot confusion matrix
    cm = get_confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, figsize=(12, 10), fontsize=8)
    
    # Plot sample predictions
    plot_sample_predictions(
        model,
        dataloaders['test'],
        class_names,
        device,
        num_samples=10
    )
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
