"""
Enhanced CNN models with attention mechanisms and residual connections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union

class ChannelAttention(nn.Module):
    """
    Channel attention module that uses both max and average pooling
    and learns channel-wise attention weights.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class EnhancedFlexibleCNN(nn.Module):
    """
    Enhanced flexible CNN architecture with customizable depth, width, 
    regularization, attention mechanisms, and residual connections.
    
    Args:
        input_channels (int): Number of channels in the input image
        num_classes (int): Number of output classes
        conv_layers (List[int]): List of filter counts for each conv layer
        fc_layers (List[int]): List of neuron counts for each fully connected layer
        kernel_size (int): Size of convolutional kernels
        pool_size (int): Size of pooling windows
        dropout (Union[float, List[float]]): Dropout rate(s) for regularization
        activation (str): Activation function to use
        pooling_type (str): Type of pooling to use ('max' or 'avg')
        use_batch_norm (bool): Whether to use batch normalization
        use_residual (bool): Whether to use residual connections
        use_attention (bool): Whether to use channel attention mechanisms
    """
    def __init__(
        self, 
        input_channels: int = 3, 
        num_classes: int = 30,
        conv_layers: List[int] = [32, 64, 128, 256, 512],
        fc_layers: List[int] = [512, 256],
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: Union[float, List[float]] = 0.5,
        activation: str = 'relu',
        pooling_type: str = 'max',
        use_batch_norm: bool = True,
        use_residual: bool = False,
        use_attention: bool = False
    ):
        super(EnhancedFlexibleCNN, self).__init__()
        
        # Store configuration
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Get activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation.lower() == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Get pooling layer
        if pooling_type.lower() == 'max':
            self.pool = nn.MaxPool2d(pool_size)
        elif pooling_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(pool_size)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")
        
        # Convert dropout to list if it's a single value
        if isinstance(dropout, (int, float)):
            self.dropout = [float(dropout)] * (len(conv_layers) + len(fc_layers))
        else:
            self.dropout = dropout
            
        # Ensure dropout has the right length
        if len(self.dropout) < len(conv_layers) + len(fc_layers):
            # Extend with the last value
            last_value = self.dropout[-1]
            extension = [last_value] * (len(conv_layers) + len(fc_layers) - len(self.dropout))
            self.dropout.extend(extension)
        
        # Build the network
        self.features = self._make_feature_extractor()
        self.classifier = self._make_classifier()
    
    def _make_feature_extractor(self):
        layers = []
        in_channels = self.input_channels
        
        for i, filters in enumerate(self.conv_layers):
            if self.use_residual and i > 0:
                # Using residual block instead of plain conv for non-first layers
                layers.append(ResidualBlock(in_channels, filters))
            else:
                # Standard convolutional block
                layers.append(nn.Conv2d(in_channels, filters, self.kernel_size, padding=1))
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm2d(filters))
                layers.append(self.activation)
            
            # Add pooling after each convolutional block
            layers.append(self.pool)
            
            # Add dropout after each convolutional block
            if self.dropout[i] > 0:
                layers.append(nn.Dropout2d(self.dropout[i]))
            
            # Add attention mechanism after specific layers (e.g., after 4th layer if it exists)
            if self.use_attention and i == min(3, len(self.conv_layers) - 1):
                layers.append(ChannelAttention(filters))
            
            in_channels = filters
        
        return nn.Sequential(*layers)
    
    def _make_classifier(self):
        # Calculate the flattened feature size
        # Assuming input is 224x224, and we have len(self.conv_layers) pooling layers
        # with pool_size = 2, the spatial size after feature extraction is 224 / (2^len(self.conv_layers))
        feature_size = 224 // (2 ** len(self.conv_layers))
        
        # If feature_size gets too small (< 1), clip it to 1
        feature_size = max(1, feature_size)
        
        # The input size to the first FC layer
        in_features = self.conv_layers[-1] * feature_size * feature_size
        
        layers = []
        
        # Add fully connected layers
        for i, neurons in enumerate(self.fc_layers):
            layers.append(nn.Linear(in_features, neurons))
            layers.append(self.activation)
            
            # Add dropout
            dropout_idx = len(self.conv_layers) + i
            if dropout_idx < len(self.dropout) and self.dropout[dropout_idx] > 0:
                layers.append(nn.Dropout(self.dropout[dropout_idx]))
            
            in_features = neurons
        
        # Final classification layer
        layers.append(nn.Linear(in_features, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_enhanced_flexible_cnn(config: Dict[str, Any]) -> nn.Module:
    """
    Create an enhanced flexible CNN model from configuration dictionary.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        nn.Module: Initialized EnhancedFlexibleCNN model
    """
    model_config = config.get('model', {})
    
    # Extract model parameters from config
    input_channels = model_config.get('input_channels', 3)
    num_classes = model_config.get('num_classes', 30)
    conv_layers = model_config.get('conv_layers', [32, 64, 128, 256, 512])
    fc_layers = model_config.get('fc_layers', [512, 256])
    kernel_size = model_config.get('kernel_size', 3)
    pool_size = model_config.get('pool_size', 2)
    dropout = model_config.get('dropout', 0.5)
    activation = model_config.get('activation', 'relu')
    pooling_type = model_config.get('pooling_type', 'max')
    use_batch_norm = model_config.get('use_batch_norm', True)
    use_residual = model_config.get('use_residual', False)
    use_attention = model_config.get('use_attention', False)
    
    # Create the model
    model = EnhancedFlexibleCNN(
        input_channels=input_channels,
        num_classes=num_classes,
        conv_layers=conv_layers,
        fc_layers=fc_layers,
        kernel_size=kernel_size,
        pool_size=pool_size,
        dropout=dropout,
        activation=activation,
        pooling_type=pooling_type,
        use_batch_norm=use_batch_norm,
        use_residual=use_residual,
        use_attention=use_attention
    )
    
    return model
