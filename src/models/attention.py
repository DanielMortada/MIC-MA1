"""
Advanced neural network components for improving CNN architectures.
Includes attention mechanisms and residual connections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel attention module to focus on important feature channels.
    
    Based on the Squeeze-and-Excitation (SE) network concept, this module 
    adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Initialize the channel attention module.
        
        Args:
            in_channels (int): Number of input channels
            reduction_ratio (int): Reduction ratio for the bottleneck
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through the channel attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Channel-recalibrated feature map of same shape as input
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important spatial locations.
    
    This module generates a spatial attention map that highlights
    important regions in the feature maps.
    """
    def __init__(self, kernel_size=7):
        """
        Initialize the spatial attention module.
        
        Args:
            kernel_size (int): Size of the convolutional kernel
        """
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through the spatial attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Spatially-recalibrated feature map of same shape as input
        """
        # Compute average and max along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along the channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Generate spatial attention map
        x_att = self.conv(x_cat)
        
        return self.sigmoid(x_att) * x

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines both channel and spatial attention in sequence.
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        """
        Initialize the CBAM module.
        
        Args:
            in_channels (int): Number of input channels
            reduction_ratio (int): Reduction ratio for channel attention
            kernel_size (int): Kernel size for spatial attention
        """
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Forward pass through the CBAM module.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Attention-refined feature map
        """
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with optional attention.
    
    Implements a standard residual connection with batch normalization
    and optional attention mechanism.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_attention=False):
        """
        Initialize the residual block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the first convolution
            use_attention (bool): Whether to use attention mechanism
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = ChannelAttention(out_channels)
    
    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_attention:
            out = self.attention(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out
