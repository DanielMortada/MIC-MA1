"""
Visualization utilities for plotting training progress and results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
import torch

def plot_training_history(history, figsize=(12, 5)):
    """
    Plot training and validation loss and accuracy
    
    Args:
        history (dict): Training history containing loss and accuracy data
        figsize (tuple): Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix',
                          cmap=plt.cm.Blues, figsize=(10, 8), fontsize=8):
    """
    Plot confusion matrix with options for normalization
    
    Args:
        cm (np.array): Confusion matrix
        classes (list): List of class names
        normalize (bool): Whether to normalize the confusion matrix
        title (str): Title for the plot
        cmap: Colormap to use
        figsize (tuple): Figure size
        fontsize (int): Font size for class labels
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f'Normalized {title}'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=fontsize)
    plt.yticks(tick_marks, classes, fontsize=fontsize)
    
    # Add counts as text
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=fontsize)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_sample_predictions(model, dataloader, class_mapping, device, num_images=8, 
                           title="Model Predictions", figsize=(15, 10), random_seed=42):
    """
    Visualize model predictions on sample images
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader containing images to sample from
        class_mapping: Dictionary mapping from class indices to class names
        device: Device to run model on
        num_images (int): Number of samples to display
        title (str): Title for the plot
        figsize (tuple): Figure size
        random_seed (int): Random seed for reproducibility
    """
    np.random.seed(random_seed)
    model.eval()
    
    # Get a batch of images
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Select random samples
    if num_images > len(images):
        num_images = len(images)
    
    indices = np.random.choice(len(images), size=num_images, replace=False)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images[indices].to(device))
        _, preds = torch.max(outputs, 1)
    
    # Convert to numpy for plotting
    images_np = images[indices].cpu().numpy()
    labels_np = labels[indices].cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    # Plot images with predictions
    fig, axes = plt.subplots(2, num_images // 2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_images):
        # Transpose image from [C, H, W] to [H, W, C]
        img = np.transpose(images_np[i], (1, 2, 0))
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Get class names
        true_class = class_mapping[labels_np[i]]
        pred_class = class_mapping[preds_np[i]]
        
        # Display image
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {pred_class}\nTrue: {true_class}")
        axes[i].axis('off')
        
        # Add color to title based on correct/incorrect prediction
        if preds_np[i] == labels_np[i]:
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_sample_images(dataset, class_mapping, num_images=5, title="Sample Images", figsize=(15, 3), random_seed=42):
    """
    Display sample images from a dataset with their class labels.
    
    Args:
        dataset: PyTorch Dataset object
        class_mapping: Dictionary mapping from class indices to class names
        num_images (int): Number of images to display
        title (str): Plot title
        figsize (tuple): Figure size
        random_seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create subplot
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    # Ensure axes is always a numpy array (even when num_images=1)
    if num_images == 1:
        axes = np.array([axes])
        
    for i in range(num_images):
        # Get a random sample
        idx = np.random.randint(0, len(dataset))
        img, label = dataset[idx]
        
        # Convert from tensor format [C, H, W] to image format [H, W, C]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            
            # Denormalize the image for display if needed
            # Assuming normalization with ImageNet mean and std
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
        
        # Get class name from the mapping
        class_name = class_mapping[label] if isinstance(label, int) else class_mapping[label.item()]
        
        # Display the image
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {class_name}")
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_class_distribution(dataloader, class_names, figsize=(12, 6)):
    """
    Plot the distribution of classes in a dataset
    
    Args:
        dataloader: PyTorch DataLoader
        class_names (list): List of class names
        figsize (tuple): Figure size
    """
    # Count instances of each class
    class_counts = {}
    for _, labels in dataloader:
        for label in labels:
            label_idx = label.item()
            if label_idx in class_counts:
                class_counts[label_idx] += 1
            else:
                class_counts[label_idx] = 1
    
    # Sort by class index
    sorted_counts = sorted(class_counts.items())
    indices = [x[0] for x in sorted_counts]
    counts = [x[1] for x in sorted_counts]
    
    # Get class names
    if class_names is not None:
        labels = [class_names[i] for i in indices]
    else:
        labels = [f"Class {i}" for i in indices]
    
    plt.figure(figsize=figsize)
    plt.bar(range(len(counts)), counts, tick_label=labels)
    plt.xticks(rotation=90)
    plt.xlabel('Class')
    plt.ylabel('Number of instances')
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.show()

def plot_sample_images(dataset, class_mapping, num_images=5, title="Sample Images", figsize=(15, 3), random_seed=42):
    """
    Display sample images from a dataset with their class labels.
    
    Args:
        dataset: PyTorch Dataset object
        class_mapping: Dictionary mapping from class indices to class names
        num_images (int): Number of images to display
        title (str): Plot title
        figsize (tuple): Figure size
        random_seed (int): Random seed for reproducibility
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create subplot
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    # Ensure axes is always a numpy array (even when num_images=1)
    if num_images == 1:
        axes = np.array([axes])
        
    for i in range(num_images):
        # Get a random sample
        idx = np.random.randint(0, len(dataset))
        img, label = dataset[idx]
        
        # Convert from tensor format [C, H, W] to image format [H, W, C]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            
            # Denormalize the image for display if needed
            # Assuming normalization with ImageNet mean and std
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
        
        # Get class name from the mapping
        class_name = class_mapping[label] if isinstance(label, int) else class_mapping[label.item()]
        
        # Display the image
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {class_name}")
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_model_layers(model, input_tensor, layer_names=None, figsize=(15, 10)):
    """
    Visualize activations of intermediate layers in a model
    
    Note: This function requires model hooks to be set up. It's a simplified version.
    For complete functionality, hooks need to be registered to the specified layers.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor to pass through the model
        layer_names (list): List of layer names to visualize (if None, will try to visualize conv layers)
        figsize (tuple): Figure size
    """
    print("Note: This is a placeholder function. To use it effectively, you'll need to register hooks to the model layers.")
    print("Please refer to the PyTorch documentation on hooks for more information.")
