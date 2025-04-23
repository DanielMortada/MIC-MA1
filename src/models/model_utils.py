"""
Model utilities for managing models across the project.
"""
import torch
import yaml
import os
import sys
from pathlib import Path
# Import project modules
from src.models.baseline import get_resnet18_model
from src.models.custom_cnn import create_custom_cnn

# Add project root to path to ensure imports work correctly
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_model_from_config(config_path, device='cpu'):
    """
    Create a model based on a configuration file
    
    Args:
        config_path (str): Path to the YAML configuration file
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        model (nn.Module): PyTorch model
    """
    # Load the configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract model configuration
    model_config = config['model']
    
    # Create the model based on the architecture
    model_name = model_config.get('name', '').lower()
    
    if model_name == 'resnet18':
        num_classes = model_config.get('num_classes', 30)
        pretrained = model_config.get('pretrained', True)
        feature_extracting = model_config.get('feature_extracting', True)
        
        model = get_resnet18_model(
            num_classes=num_classes, 
            pretrained=pretrained, 
            feature_extracting=feature_extracting
        )
        
    elif model_name == 'custom_cnn':
        num_classes = model_config.get('num_classes', 30)
        input_channels = model_config.get('input_channels', 3)
        model = create_custom_cnn(
            num_classes=num_classes,
            input_channels=input_channels
        )
        
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    # Move model to the appropriate device
    model = model.to(device)
    
    return model, config

def save_model(model, config, save_path, metrics=None, epoch=None):
    """
    Save model weights, configuration, and metrics
    
    Args:
        model (nn.Module): PyTorch model
        config (dict): Configuration dictionary
        save_path (str): Path to save the model
        metrics (dict): Optional metrics to save with the model
        epoch (int): Optional epoch number
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
        'epoch': epoch
    }
    
    # Save the model
    torch.save(checkpoint, save_path)
    
    print(f"Model saved to {save_path}")

def load_saved_model(model_path, device='cpu'):
    """
    Load a saved model
    
    Args:
        model_path (str): Path to the saved model
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        model (nn.Module): Loaded PyTorch model
        config (dict): Model configuration
        metrics (dict): Model metrics
        epoch (int): Epoch number
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract components
    model_state_dict = checkpoint['model_state_dict']
    config = checkpoint['config']
    metrics = checkpoint.get('metrics', None)
    epoch = checkpoint.get('epoch', None)
    
    # Create a new model based on the configuration
    model, _ = load_model_from_config(config)
    
    # Load the weights
    model.load_state_dict(model_state_dict)
    
    # Move model to the appropriate device
    model = model.to(device)
    
    return model, config, metrics, epoch
