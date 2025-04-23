import torch
import torchvision
import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt

print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Numpy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"PIL version: {PIL.__version__}")

# Test PyTorch CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")