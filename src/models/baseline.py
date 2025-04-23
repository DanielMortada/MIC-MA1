"""
ResNet18 baseline model for musical instrument classification.
This module provides a pre-trained ResNet18 model with a customized 
classifier head for musical instrument classification.
"""
import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes=30, pretrained=True, feature_extracting=True):
    """
    Create a ResNet-18 model with transfer learning
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        feature_extracting (bool): If True, freeze all but the final layer
        
    Returns:
        model (nn.Module): ResNet18 model
    """
    # Initialize model with or without pretrained weights
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Freeze parameters if we're just using it as a feature extractor
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def count_parameters(model):
    """
    Count the total and trainable parameters in a model
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        dict: Dictionary containing parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params
    }

def unfreeze_layers(model, layers_to_unfreeze=None):
    """
    Unfreeze specific layers in a model for fine-tuning
    
    Args:
        model (nn.Module): PyTorch model, typically a ResNet
        layers_to_unfreeze (list): List of layer names to unfreeze, e.g., ['layer4', 'fc']
        
    Returns:
        model (nn.Module): Model with specified layers unfrozen
        params_to_update (list): List of parameters that require gradients
    """
    if layers_to_unfreeze is None:
        layers_to_unfreeze = ['fc']  # By default only unfreeze the classifier
    
    params_to_update = []
    
    for name, param in model.named_parameters():
        param.requires_grad = False  # Freeze by default
        
        # Unfreeze if parameter name contains any of the specified layers
        for layer in layers_to_unfreeze:
            if layer in name:
                param.requires_grad = True
                params_to_update.append(param)
                print(f"Unfreezing: {name}")
                break
    
    return model, params_to_update
