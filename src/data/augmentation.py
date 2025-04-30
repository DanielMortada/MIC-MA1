"""
Data augmentation utilities for musical instrument classification.
"""
import torch
import torchvision.transforms as transforms
import random
import numpy as np

class AdvancedAugmentation:
    """
    Advanced data augmentation techniques for musical instrument images.
    Provides more sophisticated augmentation beyond the basic transforms.
    """
    
    @staticmethod
    def get_advanced_transforms(img_size=224, augmentation_strength='medium'):
        """
        Create advanced preprocessing transforms with configurable augmentation strength.
        
        Args:
            img_size (int): Size to resize images to (both height and width)
            augmentation_strength (str): Level of augmentation - 'light', 'medium', 'strong', or 'optimized'
            
        Returns:
            dict: Dictionary containing 'train' and 'val' transforms
        """
        # Configure augmentation parameters based on strength
        if augmentation_strength == 'light':
            rotation_degrees = 10
            jitter_params = {
                'brightness': 0.1, 
                'contrast': 0.1, 
                'saturation': 0.1,
                'hue': 0.05
            }
            use_affine = False
            random_erasing_prob = 0.1
            use_random_crop = False
            
        elif augmentation_strength == 'medium':
            rotation_degrees = 15
            jitter_params = {
                'brightness': 0.2, 
                'contrast': 0.2, 
                'saturation': 0.2,
                'hue': 0.1
            }
            use_affine = True
            random_erasing_prob = 0.2
            use_random_crop = False
            
        elif augmentation_strength == 'strong':
            rotation_degrees = 30
            jitter_params = {
                'brightness': 0.3, 
                'contrast': 0.3, 
                'saturation': 0.3,
                'hue': 0.15
            }
            use_affine = True
            random_erasing_prob = 0.3
            use_random_crop = False
            
        elif augmentation_strength == 'optimized':
            # Optimized settings specifically for Deeper CNN
            rotation_degrees = 20
            jitter_params = {
                'brightness': 0.2, 
                'contrast': 0.2, 
                'saturation': 0.2,
                'hue': 0.1
            }
            use_affine = True
            random_erasing_prob = 0.3
            use_random_crop = True
            
        else:
            raise ValueError("augmentation_strength must be 'light', 'medium', 'strong', or 'optimized'")
          
        # Base list of transforms
        train_transform_list = []
        
        # Add RandomResizedCrop for optimized augmentation
        if augmentation_strength == 'optimized' and use_random_crop:
            train_transform_list.append(
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0))
            )
        else:
            train_transform_list.append(
                transforms.Resize((img_size, img_size))
            )
            
        # Add other transforms
        train_transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(rotation_degrees),
            transforms.ColorJitter(**jitter_params),
        ])
        
        # Add affine transform if specified
        if use_affine:
            train_transform_list.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=None
                )
            )
            
        # Add ToTensor and Normalize
        train_transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Add random erasing as final augmentation
        if random_erasing_prob > 0:
            train_transform_list.append(
                transforms.RandomErasing(p=random_erasing_prob)
            )
            
        # Create the final transforms
        train_transform = transforms.Compose(train_transform_list)
        
        # Validation transform (simpler, no augmentations)
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return {
            'train': train_transform,
            'val': val_transform
        }
