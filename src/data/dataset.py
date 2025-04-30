"""
Dataset management for musical instrument classification.
"""
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class InstrumentDataset(Dataset):
    """
    Dataset class for loading musical instrument images.
    Handles both file paths and class labels.
    """
    def __init__(self, image_paths, classes, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of image file paths
            classes (list): List of class names
            transform: Optional transform to be applied to the images
        """
        self.image_paths = image_paths
        self.transform = transform
        self.classes = classes
        
        # Extract labels from file paths
        self.labels = []
        for path in image_paths:
            class_name = os.path.basename(os.path.dirname(path))
            self.labels.append(classes.index(class_name))
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is the transformed image and label is the class index
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(img_size=224, augmentation_level='standard'):
    """
    Get standard transforms for training and validation.
    
    Args:
        img_size (int): Size to resize images to
        augmentation_level (str): Level of augmentation ('minimal', 'standard', 'advanced')
        
    Returns:
        dict: Dictionary containing 'train' and 'val' transforms
    """
    # Basic validation transform (resize and normalize)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Configure training transforms based on augmentation level
    if augmentation_level == 'minimal':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif augmentation_level == 'standard':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif augmentation_level == 'advanced':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
    else:
        raise ValueError("augmentation_level must be 'minimal', 'standard', or 'advanced'")
    
    return {
        'train': train_transform,
        'val': val_transform
    }
