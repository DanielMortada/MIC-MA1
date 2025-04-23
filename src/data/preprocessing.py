"""
Data preprocessing utilities for musical instrument classification.
"""
import torch
import torchvision.transforms as transforms

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
