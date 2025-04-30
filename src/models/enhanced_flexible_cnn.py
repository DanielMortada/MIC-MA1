"""
Enhanced Flexible CNN module with advanced architectural features.
"""
import torch
import torch.nn as nn
from .flexible_cnn import FlexibleCNN
from .attention import ChannelAttention, ResidualBlock

class EnhancedFlexibleCNN(nn.Module):
    """
    Enhanced version of FlexibleCNN with added attention mechanisms
    and residual connections for better performance.
    """
    def __init__(self, in_channels=3, num_classes=30, conv_layers=None, fc_layers=None,
                 kernel_size=3, pool_size=2, dropout=None, activation='relu',
                 pooling_type='max', use_batch_norm=True, use_residual=False,
                 use_attention=False):
        """
        Initialize the enhanced flexible CNN.
        
        Args:
            in_channels (int): Number of input channels (3 for RGB images)
            num_classes (int): Number of output classes
            conv_layers (list): List of integers representing the number of filters in each conv layer
            fc_layers (list): List of integers representing the size of each fully connected layer
            kernel_size (int): Size of the convolutional kernels
            pool_size (int): Size of the pooling windows
            dropout (list or float): Dropout rates for each layer
            activation (str): Activation function to use ('relu', 'leaky_relu', etc.)
            pooling_type (str): Type of pooling to use ('max' or 'avg')
            use_batch_norm (bool): Whether to use batch normalization
            use_residual (bool): Whether to use residual connections
            use_attention (bool): Whether to use attention mechanisms
        """
        super(EnhancedFlexibleCNN, self).__init__()
        
        # Set default values
        if conv_layers is None:
            conv_layers = [32, 64, 128, 256]
        
        if fc_layers is None:
            fc_layers = [512, 256]
        
        # Handle dropout
        if dropout is None:
            dropout = [0.1] * len(conv_layers) + [0.5] * len(fc_layers)
        elif isinstance(dropout, (int, float)):
            dropout = [dropout] * (len(conv_layers) + len(fc_layers))
        
        # Store configuration
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout = dropout
        self.activation_name = activation
        self.pooling_type = pooling_type
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Create activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create pooling function
        if pooling_type == 'max':
            self.pool = nn.MaxPool2d(pool_size)
        elif pooling_type == 'avg':
            self.pool = nn.AvgPool2d(pool_size)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")
        
        # Create convolutional layers
        self.conv_blocks = nn.ModuleList()
        
        # First convolutional layer
        if use_residual:
            self.conv_blocks.append(
                ResidualBlock(in_channels, conv_layers[0], use_attention=use_attention and (0 % 2 == 1))
            )
        else:
            block = []
            block.append(nn.Conv2d(in_channels, conv_layers[0], kernel_size=kernel_size, padding=kernel_size//2))
            if use_batch_norm:
                block.append(nn.BatchNorm2d(conv_layers[0]))
            block.append(self.activation)
            if dropout[0] > 0:
                block.append(nn.Dropout2d(dropout[0]))
            self.conv_blocks.append(nn.Sequential(*block))
        
        # Add attention module after first block if specified
        if use_attention and not use_residual and 0 % 2 == 1:
            self.conv_blocks.append(ChannelAttention(conv_layers[0]))
        
        # Add pooling layer
        self.conv_blocks.append(self.pool)
        
        # Remaining convolutional layers
        for i in range(1, len(conv_layers)):
            if use_residual:
                self.conv_blocks.append(
                    ResidualBlock(conv_layers[i-1], conv_layers[i], use_attention=use_attention and (i % 2 == 1))
                )
            else:
                block = []
                block.append(nn.Conv2d(conv_layers[i-1], conv_layers[i], kernel_size=kernel_size, padding=kernel_size//2))
                if use_batch_norm:
                    block.append(nn.BatchNorm2d(conv_layers[i]))
                block.append(self.activation)
                if dropout[i] > 0:
                    block.append(nn.Dropout2d(dropout[i]))
                self.conv_blocks.append(nn.Sequential(*block))
            
            # Add attention module after block if specified and not already included in ResidualBlock
            if use_attention and not use_residual and i % 2 == 1:
                self.conv_blocks.append(ChannelAttention(conv_layers[i]))
                
            # Add pooling layer
            self.conv_blocks.append(self.pool)
        
        # Calculate size of flattened features
        # Assuming starting with 224x224 image, each pooling divides by 2
        feature_size = 224 // (2 ** len(conv_layers))
        flattened_size = conv_layers[-1] * feature_size * feature_size
        
        # Create fully connected layers
        self.fc_blocks = nn.ModuleList()
        
        # First FC layer
        fc_block = []
        fc_block.append(nn.Linear(flattened_size, fc_layers[0]))
        fc_block.append(self.activation)
        if len(dropout) > len(conv_layers):
            dropout_rate = dropout[len(conv_layers)]
            if dropout_rate > 0:
                fc_block.append(nn.Dropout(dropout_rate))
        self.fc_blocks.append(nn.Sequential(*fc_block))
        
        # Remaining FC layers
        for i in range(1, len(fc_layers)):
            fc_block = []
            fc_block.append(nn.Linear(fc_layers[i-1], fc_layers[i]))
            fc_block.append(self.activation)
            if len(dropout) > len(conv_layers) + i:
                dropout_rate = dropout[len(conv_layers) + i]
                if dropout_rate > 0:
                    fc_block.append(nn.Dropout(dropout_rate))
            self.fc_blocks.append(nn.Sequential(*fc_block))
        
        # Final classification layer
        self.classifier = nn.Linear(fc_layers[-1], num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, num_classes]
        """
        # Forward through convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Flatten features
        x = x.view(x.size(0), -1)
        
        # Forward through fully connected blocks
        for block in self.fc_blocks:
            x = block(x)
        
        # Final classification
        return self.classifier(x)

def create_enhanced_flexible_cnn(config):
    """
    Create an enhanced flexible CNN model from configuration.
    
    Args:
        config (dict): Configuration dictionary for the model
        
    Returns:
        EnhancedFlexibleCNN: Initialized model
    """
    model_config = config.get('model', {})
    
    # Extract parameters from config
    in_channels = model_config.get('input_channels', 3)
    num_classes = model_config.get('num_classes', 30)
    conv_layers = model_config.get('conv_layers', [32, 64, 128, 256])
    fc_layers = model_config.get('fc_layers', [512, 256])
    kernel_size = model_config.get('kernel_size', 3)
    pool_size = model_config.get('pool_size', 2)
    dropout = model_config.get('dropout', None)
    activation = model_config.get('activation', 'relu')
    pooling_type = model_config.get('pooling_type', 'max')
    use_batch_norm = model_config.get('use_batch_norm', True)
    use_residual = model_config.get('use_residual', False)
    use_attention = model_config.get('use_attention', False)
    
    # Create model
    model = EnhancedFlexibleCNN(
        in_channels=in_channels,
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
