"""
Google Colab integration utilities for the Musical Instrument Classification project.
This script provides utilities for running your project on Google Colab using a cloned GitHub repository.
No path modifications or Google Drive mounting is needed since we're using the cloned repo directly.
"""

import os
import subprocess
import torch
import importlib.util  # Used for checking if modules are available

def setup_colab_environment():
    """
    Set up the Google Colab environment for the project.
    Installs necessary dependencies for the cloned repository.
    """
    print("Setting up Colab environment for Musical Instrument Classification...")
    
    # Check if running in Colab using importlib
    is_colab = importlib.util.find_spec("google.colab") is not None
    
    if not is_colab:
        print("Not running in Google Colab environment")
        return False
        
    print("Running in Google Colab environment")
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run(["pip", "install", "torch", "torchvision", "tqdm", "matplotlib", "seaborn", "scikit-learn", "pyyaml"], check=True)
    
    print("Environment setup complete!")
    return True

def check_gpu():
    """
    Check for GPU availability and return device information.
    
    Returns:
        device: PyTorch device (cuda or cpu)
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("No GPU detected, using CPU. Training will be slower.")
    
    return device

if __name__ == "__main__":
    # Example usage
    print("This script is intended to be imported, not run directly.")
    print("Example usage in a Colab notebook:")
    print("```python")
    print("from scripts.colab_integration import setup_colab_environment, check_gpu")
    print("setup_colab_environment()  # Install dependencies")
    print("device = check_gpu()       # Check for GPU availability")
    print("# Now use device in your training code to leverage GPU acceleration")
    print("```")

def check_gpu():
    """
    Check for GPU availability and return device information.
    
    Returns:
        device: PyTorch device (cuda or cpu)
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("No GPU detected, using CPU. Training will be slower.")
    
    return device

if __name__ == "__main__":
    # Example usage
    print("This script is intended to be imported, not run directly.")
    print("Example usage in a Colab notebook:")
    print("```python")
    print("from scripts.colab_integration import setup_colab_environment, check_gpu")
    print("setup_colab_environment()  # Install dependencies")
    print("device = check_gpu()       # Check for GPU availability")
    print("# Now use device in your training code to leverage GPU acceleration")
    print("```")
