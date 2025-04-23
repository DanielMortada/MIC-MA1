"""
Data loading utilities for musical instrument classification.
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def load_datasets(data_dir, transforms, batch_size=32, num_workers=0, pin_memory=False):
    """
    Load musical instruments datasets from the specified directory.
    
    Args:
        data_dir (str): Root directory containing train, valid, and test folders
        transforms (dict): Dictionary with 'train', 'val', and 'test' transforms
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        pin_memory (bool): Whether to pin memory (beneficial for GPU training)
        
    Returns:
        dict: Dictionary containing dataset and dataloader objects
    """
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")
    
    # Verify data paths
    for dir_path, dir_name in [(train_dir, "Training"), (valid_dir, "Validation"), (test_dir, "Test")]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_name} data directory not found: {dir_path}")
    
    # Load datasets
    train_dataset = ImageFolder(train_dir, transform=transforms['train'])
    valid_dataset = ImageFolder(valid_dir, transform=transforms['val'])
    test_dataset = ImageFolder(test_dir, transform=transforms['test'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Create mapping dictionaries
    idx_to_class = {i: c for i, c in enumerate(train_dataset.classes)}
    class_to_idx = {c: i for i, c in enumerate(train_dataset.classes)}
    
    return {
        'datasets': {
            'train': train_dataset,
            'val': valid_dataset,
            'test': test_dataset
        },
        'dataloaders': {
            'train': train_loader,
            'val': valid_loader,
            'test': test_loader
        },
        'class_mappings': {
            'idx_to_class': idx_to_class,
            'class_to_idx': class_to_idx
        },
        'num_classes': len(train_dataset.classes)
    }
