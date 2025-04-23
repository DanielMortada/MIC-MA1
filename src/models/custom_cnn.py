"""
Custom CNN architectures for musical instrument classification.
"""
import torch.nn as nn
import torch.nn.functional as F

class MusicInstrumentCNN(nn.Module):
    """
    A custom CNN architecture specifically designed for musical instrument classification.
    Includes 5 convolutional blocks with progressive feature extraction and global average pooling.
    """
    def __init__(self, num_classes=30, input_channels=3):
        """
        Custom CNN for musical instrument classification
        
        Args:
            num_classes: Number of output classes (default: 30 for musical instruments)
            input_channels: Number of input image channels (default: 3 for RGB)
        """
        super(MusicInstrumentCNN, self).__init__()
        
        # Block 1: Input shape (3, 224, 224)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # Output: (32, 112, 112)
        self.dropout1 = nn.Dropout(0.1)
        
        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # Output: (64, 56, 56)
        self.dropout2 = nn.Dropout(0.2)
        
        # Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # Output: (128, 28, 28)
        self.dropout3 = nn.Dropout(0.3)
        
        # Block 4
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)  # Output: (256, 14, 14)
        self.dropout4 = nn.Dropout(0.4)
        
        # Block 5 - Added a 5th block to handle 224x224 input
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2) # Output: (512, 7, 7)
        self.dropout5 = nn.Dropout(0.5)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Output: (512, 1, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 512)
        self.dropout6 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        # Block 5
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.pool5(x)
        x = self.dropout5(x)
        
        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout6(x)
        x = self.fc2(x)
        
        return x


def create_custom_cnn(num_classes=30, input_channels=3):
    """
    Factory function to create a custom CNN model
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (3 for RGB images)
        
    Returns:
        model (nn.Module): Custom CNN model
    """
    model = MusicInstrumentCNN(
        num_classes=num_classes,
        input_channels=input_channels
    )
    
    return model