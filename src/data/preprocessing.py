"""
Data preprocessing utilities for musical instrument classification.
"""
import torch
import torchvision.transforms as transforms
import os
import random
from pathlib import Path

def get_preprocessing_transforms(img_size=224):
    """
    Create standard preprocessing transforms for training, validation, and testing.
    
    Args:
        img_size (int): Size to resize images to (both height and width)
        
    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' transforms
    """
    # Image preprocessing and data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transforms,
        'val': val_test_transforms,
        'test': val_test_transforms
    }

def create_train_val_split(data_dir, val_split=0.2, test_split=0.0, seed=None):
    """
    Create train, validation, and optionally test splits from a directory of images.
    
    Args:
        data_dir (str or Path): Directory containing subdirectories for each class
        val_split (float): Proportion of data to use for validation (0.0 to 1.0)
        test_split (float): Proportion of data to use for testing (0.0 to 1.0)
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_files, val_files, classes) or (train_files, val_files, test_files, classes)
               depending on whether test_split > 0
    """
    if seed is not None:
        random.seed(seed)
    
    data_dir = Path(data_dir) if not isinstance(data_dir, Path) else data_dir
    
    # Get class names (subdirectory names)
    classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    classes.sort()  # Sort alphabetically for consistency
    
    # Collect all file paths
    all_files = []
    for class_name in classes:
        class_dir = data_dir / class_name
        for img_path in class_dir.glob('*.*'):
            # Skip non-image files
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                all_files.append(str(img_path))
    
    # Shuffle files
    random.shuffle(all_files)
    
    # Calculate split sizes
    total_size = len(all_files)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size
    
    # Split the files
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    
    if test_split > 0:
        test_files = all_files[train_size + val_size:]
        return train_files, val_files, test_files, classes
    else:
        return train_files, val_files, classes
