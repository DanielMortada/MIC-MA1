"""
Google Colab integration utilities for the Musical Instrument Classification project.
This script provides utilities for running your manually uploaded project on Google Colab.
"""

import os
import subprocess
import torch
import importlib.util  # Used for checking if modules are available

def setup_colab_environment():
    """
    Set up the Google Colab environment for the project.
    Installs necessary dependencies but assumes manual project upload.
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
    
    # Note: No repository cloning since project will be uploaded manually by the user
    
    print("Environment setup complete!")
    return True

def mount_google_drive():
    """
    Mount Google Drive to access data and save results.
    """
    try:
        # Only import if running in Colab
        if importlib.util.find_spec("google.colab") is not None:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted at /content/drive")
            return True
        else:
            print("Not running in Google Colab environment")
            return False
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
        return False

def copy_data_from_drive(drive_data_path, local_data_path="data/processed"):
    """
    Copy dataset from Google Drive to Colab VM.
    
    Args:
        drive_data_path (str): Path to the data in Google Drive
        local_data_path (str): Local path where data should be copied
    """
    if not os.path.exists("/content/drive"):
        print("Google Drive not mounted. Please run mount_google_drive() first.")
        return False
    
    print(f"Copying data from {drive_data_path} to {local_data_path}...")
    
    # Ensure the target directory exists
    os.makedirs(local_data_path, exist_ok=True)
    
    # Copy data using cp command
    result = subprocess.run(
        ["cp", "-r", drive_data_path, local_data_path],
        capture_output=True
    )
    
    if result.returncode == 0:
        print("Data copied successfully!")
        return True
    else:
        print("Failed to copy data:")
        print(result.stderr.decode())
        return False

def save_results_to_drive(local_results_path, drive_results_path):
    """
    Save experiment results to Google Drive.
    
    Args:
        local_results_path (str): Path to local results
        drive_results_path (str): Path in Google Drive to save results
    """
    if not os.path.exists("/content/drive"):
        print("Google Drive not mounted. Please run mount_google_drive() first.")
        return False
    
    print(f"Copying results from {local_results_path} to {drive_results_path}...")
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(drive_results_path), exist_ok=True)
    
    # Copy data using cp command
    result = subprocess.run(
        ["cp", "-r", local_results_path, drive_results_path],
        capture_output=True
    )
    
    if result.returncode == 0:
        print("Results saved to Drive successfully!")
        return True
    else:
        print("Failed to save results:")
        print(result.stderr.decode())
        return False

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
