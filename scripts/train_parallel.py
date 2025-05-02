#!/usr/bin/env python
"""
Parallel training script for musical instrument classification.
This script trains multiple model architectures in parallel using the flexible framework.

Usage: python scripts/train_parallel.py
"""
import os
import sys
import yaml
import time
import argparse
import concurrent.futures
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Add project root to path to ensure imports work correctly
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

# Import project modules
from src.data.preprocessing import get_preprocessing_transforms
from src.data.augmentation import AdvancedAugmentation
from src.data.dataloader import load_datasets
from src.models.baseline import get_resnet18_model, unfreeze_layers
from src.models.custom_cnn import create_custom_cnn
from src.models.flexible_cnn import create_flexible_cnn
from src.training.trainer import train_model, evaluate_model
from src.training.metrics import compute_metrics, get_confusion_matrix
from src.visualization.plotting import plot_training_history, plot_confusion_matrix
from src.models.model_utils import save_model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Parallel Training Framework")
    parser.add_argument(
        "--configs",
        nargs='+',
        default=[
            "config/resnet18_baseline.yaml",
            "config/custom_cnn_base.yaml",
            "config/custom_cnn_deeper.yaml",
            "config/custom_cnn_wider.yaml",
            "config/custom_cnn_regularized.yaml"
        ],
        help="List of configuration files to use for training"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Maximum number of parallel training processes"
    )
    return parser.parse_args()

def train_model(config_path):
    """
    Train a model using the specified configuration file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Training results
    """
    print(f"Starting training with config: {config_path}")
    
    start_time = time.time()
    
    # Build the command to run train_flexible.py with the specified config
    cmd = [sys.executable, os.path.join(project_root, "scripts", "train_flexible.py"), "--config", config_path]
    
    # Run the command and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    training_time = time.time() - start_time
    
    if process.returncode != 0:
        print(f"Error training model with config {config_path}")
        print(f"Error: {stderr}")
        return {
            "config": config_path,
            "status": "error",
            "error": stderr,
            "training_time": training_time
        }
    
    # Parse stdout to extract metrics
    test_accuracy = None
    f1_score = None
    model_name = config_path.split("/")[-1].replace(".yaml", "")
    
    for line in stdout.split("\n"):
        if "Test Accuracy:" in line:
            try:
                test_accuracy = float(line.split(":")[1].strip())
            except:
                pass
        elif "Macro-average F1 Score:" in line:
            try:
                f1_score = float(line.split(":")[1].strip())
            except:
                pass
    
    print(f"Completed training with config: {config_path}")
    print(f"Test Accuracy: {test_accuracy}, F1 Score: {f1_score}, Training Time: {training_time:.2f}s")
    
    return {
        "config": config_path,
        "model_name": model_name,
        "status": "success",
        "test_accuracy": test_accuracy,
        "f1_score": f1_score,
        "training_time": training_time
    }

def plot_comparison(results):
    """
    Plot comparison of model performance
    
    Args:
        results (list): List of model training results
        
    Returns:
        pd.DataFrame: Sorted comparison table
    """
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Check if there are any successful results
    if df.empty or 'test_accuracy' not in df.columns:
        print("No valid results to plot comparison")
        return pd.DataFrame()
    
    # Sort by test accuracy
    df = df.sort_values("test_accuracy", ascending=False)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot test accuracy
    model_names = df["model_name"].tolist()
    x = np.arange(len(model_names))
    
    # First subplot - Accuracy and F1 Score
    bar_width = 0.35
    ax1.bar(x - bar_width/2, df["test_accuracy"], bar_width, label="Accuracy")
    ax1.bar(x + bar_width/2, df["f1_score"], bar_width, label="F1 Score")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Score")
    ax1.set_title("Test Accuracy and F1 Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Second subplot - Training Time
    ax2.bar(model_names, df["training_time"] / 60, color="green")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Training Time (minutes)")
    ax2.set_title("Training Time")
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(project_root, "experiments", f"comparison_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plot
    plt.savefig(os.path.join(save_dir, "model_comparison.png"))
    print(f"Comparison plot saved to {save_dir}/model_comparison.png")
    
    # Also save a CSV with the results
    df.to_csv(os.path.join(save_dir, "model_comparison.csv"), index=False)
    print(f"Comparison data saved to {save_dir}/model_comparison.csv")
    
    # Create a more readable summary table
    summary_df = df[['model_name', 'test_accuracy', 'f1_score', 'training_time']].copy()
    summary_df['training_time'] = summary_df['training_time'] / 60
    summary_df.columns = ['Model', 'Test Accuracy', 'F1 Score', 'Training Time (minutes)']
    summary_df['Test Accuracy'] = summary_df['Test Accuracy'].apply(lambda x: f"{x:.4f}")
    summary_df['F1 Score'] = summary_df['F1 Score'].apply(lambda x: f"{x:.4f}")
    summary_df['Training Time (minutes)'] = summary_df['Training Time (minutes)'].apply(lambda x: f"{x:.2f}")
      # Print the summary table
    print("\nModel Performance Comparison:")
    print(summary_df.to_string(index=False))
    
    plt.close(fig)
    
    # Save results to plots_dir as well
    plots_dir = os.path.join(project_root, "results")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "model_comparison.png"), dpi=300, bbox_inches="tight")
    
    return summary_df

def main():
    # Parse arguments
    args = parse_args()
    
    print(f"Starting parallel training with {len(args.configs)} configurations")
    print(f"Using a maximum of {args.max_workers} parallel workers")
    
    results = []
    
    # Train models in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Start training for all configurations
        future_to_config = {executor.submit(train_model, config): config for config in args.configs}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error training model with config {config}: {e}")
                results.append({
                    "config": config,
                    "model_name": config.split("/")[-1].replace(".yaml", ""),
                    "status": "error",
                    "error": str(e),
                    "test_accuracy": None,
                    "f1_score": None,
                    "training_time": None
                })
    
    # Plot comparison of model performance
    print("\nAll training complete! Generating comparison plots...")
    comparison_table = plot_comparison([r for r in results if r["status"] == "success"])
    
    print("\nParallel training complete!")
    
    # Suggest the best model
    best_model = comparison_table.iloc[0]["Model"]
    print(f"\nBased on test accuracy, the best model is: {best_model}")
    
if __name__ == "__main__":
    main()
