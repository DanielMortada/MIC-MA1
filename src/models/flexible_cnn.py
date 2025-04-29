"""
Flexible CNN architecture for musical instrument classification.
This module provides a configurable CNN architecture that can be adapted through configuration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    """
    A flexible CNN architecture that can be configured with different numbers of layers,
    filter sizes, kernel sizes, pooling types, and other parameters.
    
    This allows for experimentation with different architectures without changing the code.
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=30,
        conv_layers=[64, 128, 256, 512],
        fc_layers=[512, 256],
        kernel_size=3,
        pool_size=2,
        dropout=0.5,
        activation='relu',
        pooling_type='max',
        use_batch_norm=True
    ):
        """
        Initialize a flexible CNN model.
        
        Args:
            in_channels (int): Number of input channels (3 for RGB images)
            num_classes (int): Number of output classes
            conv_layers (list): List of integers representing the number of filters in each conv layer
            fc_layers (list): List of integers representing the size of each fully connected layer
            kernel_size (int or list): Size of convolutional kernels (int or list of ints for each layer)
            pool_size (int): Size of pooling windows
            dropout (float or list): Dropout rate after each layer (float or list of floats)
            activation (str): Activation function to use ('relu', 'leaky_relu', 'elu', etc.)
            pooling_type (str): Type of pooling to use ('max' or 'avg')
            use_batch_norm (bool): Whether to use batch normalization
        """
        super(FlexibleCNN, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * len(conv_layers)
        self.dropout = dropout if isinstance(dropout, list) else [dropout] * (len(conv_layers) + len(fc_layers))
        self.activation = activation
        self.pooling_type = pooling_type
        self.use_batch_norm = use_batch_norm
        
        # Create activation function
        if activation == 'relu':
            self.act_fn = F.relu
        elif activation == 'leaky_relu':
            self.act_fn = F.leaky_relu
        elif activation == 'elu':
            self.act_fn = F.elu
        else:
            self.act_fn = F.relu
            
        # Create the convolutional layers
        self.features = self._make_conv_layers()
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Create the fully connected layers
        self.classifier = self._make_fc_layers()
    
    def _make_conv_layers(self):
        layers = []
        input_channels = self.in_channels
        
        for idx, num_filters in enumerate(self.conv_layers):            # First convolutional layer in the block
            layers.append(nn.Conv2d(
                input_channels, 
                num_filters, 
                kernel_size=self.kernel_size[idx], 
                padding=self.kernel_size[idx]//2
            ))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(num_filters))
            
            # Use the configured activation function
            if self.activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif self.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(inplace=True))
            elif self.activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            
            # Second convolutional layer in the block (same number of filters)
            layers.append(nn.Conv2d(
                num_filters, 
                num_filters, 
                kernel_size=self.kernel_size[idx], 
                padding=self.kernel_size[idx]//2
            ))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(num_filters))
            
            # Use the configured activation function
            if self.activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif self.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(inplace=True))
            elif self.activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            
            # Pooling layer
            if self.pooling_type == 'max':
                layers.append(nn.MaxPool2d(self.pool_size))
            else:  # 'avg'
                layers.append(nn.AvgPool2d(self.pool_size))
            
            # Dropout layer
            if self.dropout[idx] > 0:
                layers.append(nn.Dropout(self.dropout[idx]))
            input_channels = num_filters
            
        return nn.Sequential(*layers)
    
    def _make_fc_layers(self):
        layers = []
        
        # First FC layer takes input from flattened feature maps
        input_size = self.conv_layers[-1]  # Size after global pooling
        
        for idx, fc_size in enumerate(self.fc_layers):
            layers.append(nn.Linear(input_size, fc_size))
            
            # Use the configured activation function
            if self.activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif self.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(inplace=True))
            elif self.activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            
            if idx < len(self.fc_layers) - 1 and self.dropout[len(self.conv_layers) + idx] > 0:
                layers.append(nn.Dropout(self.dropout[len(self.conv_layers) + idx]))
            
            input_size = fc_size
        
        # Final classification layer
        layers.append(nn.Linear(input_size, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def create_flexible_cnn(config):
    """
    Factory function to create a flexible CNN model from configuration.
    
    Args:
        config (dict): Configuration dictionary with model parameters
        
    Returns:
        model (nn.Module): Flexible CNN model
    """
    model_config = config.get('model', {})
    
    return FlexibleCNN(
        in_channels=model_config.get('input_channels', 3),
        num_classes=model_config.get('num_classes', 30),
        conv_layers=model_config.get('conv_layers', [64, 128, 256, 512]),
        fc_layers=model_config.get('fc_layers', [512, 256]),
        kernel_size=model_config.get('kernel_size', 3),
        pool_size=model_config.get('pool_size', 2),
        dropout=model_config.get('dropout', 0.5),
        activation=model_config.get('activation', 'relu'),
        pooling_type=model_config.get('pooling_type', 'max'),
        use_batch_norm=model_config.get('use_batch_norm', True)
    )