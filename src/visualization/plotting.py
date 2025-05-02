"""
Visualization utilities for plotting model training and evaluation results.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple

def plot_training_history(history: Dict[str, List[float]]) -> None:
    """
    Plot the training and validation metrics.
    
    Args:
        history (Dict[str, List[float]]): Dictionary containing training history
            with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr'
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Determine subplot layout based on available data
    has_lr = 'lr' in history
    subplot_count = 3 if has_lr else 2
    
    # Create a subplot grid
    plt.figure(figsize=(16, 10))
    
    # Plot training & validation loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate if available
    if has_lr:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history['lr'], 'g-')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        # Plot validation accuracy vs. learning rate
        plt.subplot(2, 2, 4)
        plt.scatter(history['lr'], history['val_acc'], alpha=0.7)
        plt.title('Validation Accuracy vs. Learning Rate')
        plt.xlabel('Learning Rate')
        plt.ylabel('Validation Accuracy')
        plt.xscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(
    cm: np.ndarray, 
    classes: List[str], 
    normalize: bool = False, 
    figsize: Tuple[int, int] = (12, 10),
    fontsize: int = 8
) -> None:
    """
    Plot a confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        classes (List[str]): List of class names
        normalize (bool): Whether to normalize the confusion matrix
        figsize (Tuple[int, int]): Figure size
        fontsize (int): Font size for labels
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=fontsize+2)
    plt.ylabel('True', fontsize=fontsize+2)
    plt.title('Confusion Matrix', fontsize=fontsize+4)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.tight_layout()
    plt.show()

def plot_sample_predictions(
    model, 
    dataloader, 
    class_names: List[str],
    device,
    num_samples: int = 8,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot sample predictions from the model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader containing samples to visualize
        class_names (List[str]): List of class names
        device: Device to run inference on
        num_samples (int): Number of samples to visualize
        figsize (Tuple[int, int]): Figure size
    """
    import torch
    from torchvision import transforms
    
    # Set the model to evaluation mode
    model.eval()
    
    # Get a batch of data
    images, labels = next(iter(dataloader))
    
    # Only select a subset of images
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
    
    # Denormalize the images for visualization
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ])
    
    # Plot the images
    fig = plt.figure(figsize=figsize)
    
    for i in range(len(images)):
        # Denormalize and convert to numpy array
        img = denormalize(images[i])
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).cpu().numpy()
        
        # Get the prediction and true label
        pred_label = class_names[preds[i].item()]
        true_label = class_names[labels[i].item()]
        
        # Add subplot
        ax = fig.add_subplot(2, num_samples//2, i+1)
        ax.imshow(img)
        
        # Color the title based on whether prediction is correct
        title_color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_learning_rate_finder_results(
    learning_rates: List[float],
    losses: List[float],
    skip_start: int = 5,
    skip_end: int = 5
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the results of the learning rate finder.
    
    Args:
        learning_rates (List[float]): List of learning rates
        losses (List[float]): List of corresponding losses
        skip_start (int): Number of batches to skip at the start
        skip_end (int): Number of batches to skip at the end
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and axes objects
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Skip the first few batches which are often noisy and the last few
    # which are often diverging
    learning_rates = learning_rates[skip_start:-skip_end] if skip_end > 0 else learning_rates[skip_start:]
    losses = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]
    
    ax.plot(learning_rates, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Rate Finder Results')
    ax.grid(True)
    
    # Add vertical line at the learning rate with the steepest gradient
    # This is one heuristic for choosing a good learning rate
    gradients = np.gradient(losses, np.log(learning_rates))
    steepest_idx = np.argmin(gradients)
    optimal_lr = learning_rates[steepest_idx]
    
    ax.axvline(x=optimal_lr, color='r', linestyle='--', 
               label=f'Steepest Gradient: {optimal_lr:.1e}')
    
    # Add vertical line at the learning rate with the minimum loss
    # This is often too high as a starting point
    min_loss_idx = np.argmin(losses)
    min_loss_lr = learning_rates[min_loss_idx]
    
    ax.axvline(x=min_loss_lr, color='g', linestyle='--',
               label=f'Min Loss: {min_loss_lr:.1e}')
    
    # Suggest a reasonable learning rate (divide by 10)
    suggested_lr = optimal_lr / 10
    ax.axvline(x=suggested_lr, color='y', linestyle='-.',
               label=f'Suggested LR: {suggested_lr:.1e}')
    
    ax.legend()
    plt.tight_layout()
    
    return fig, ax

def plot_feature_maps(
    model,
    image,
    layer_idx: int = 0,
    num_filters: int = 16,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize feature maps from a specific layer of the model.
    
    Args:
        model: PyTorch model
        image: Input image tensor
        layer_idx (int): Index of the layer to visualize
        num_filters (int): Number of filters to visualize
        figsize (Tuple[int, int]): Figure size
    """
    import torch
    
    # Set the model to evaluation mode
    model.eval()
    
    # Get a list of all feature extractors
    feature_extractor_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            feature_extractor_layers.append(module)
    
    # Ensure valid layer index
    if layer_idx >= len(feature_extractor_layers):
        print(f"Layer index {layer_idx} out of range. Model has {len(feature_extractor_layers)} convolutional layers.")
        return
    
    # Get the feature maps
    with torch.no_grad():
        # Forward pass through the model up to the selected layer
        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # Register hook to the layer
        handle = feature_extractor_layers[layer_idx].register_forward_hook(get_activation('feature_maps'))
        
        # Forward pass
        _ = model(image.unsqueeze(0))
        
        # Get the feature maps
        feature_maps = activation['feature_maps']
        
        # Remove the hook
        handle.remove()
    
    # Get the number of filters
    num_filters_actual = min(num_filters, feature_maps.shape[1])
    
    # Plot the feature maps
    plt.figure(figsize=figsize)
    
    for i in range(num_filters_actual):
        plt.subplot(4, num_filters_actual // 4 + (1 if num_filters_actual % 4 != 0 else 0), i+1)
        plt.imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    
    plt.suptitle(f'Feature Maps - Layer {layer_idx} (Conv2d)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
