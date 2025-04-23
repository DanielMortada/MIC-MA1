"""
Training utilities for model training and evaluation.
"""
import torch
import time
import copy
from tqdm.notebook import tqdm
import numpy as np

def train_model(model, dataloaders, criterion, optimizer, device, 
                scheduler=None, num_epochs=10, verbose=True):
    """
    General-purpose training function that handles the training and validation process
    
    Args:
        model (nn.Module): PyTorch model to train
        dataloaders (dict): Dictionary of PyTorch DataLoader objects for 'train' and 'val'
        criterion: Loss function
        optimizer: Optimizer to use
        device (torch.device): Device to train on (GPU or CPU)
        scheduler: Learning rate scheduler (optional)
        num_epochs (int): Number of epochs to train for
        verbose (bool): Whether to print progress
        
    Returns:
        model: Best model based on validation accuracy
        history (dict): Training and validation metrics
        training_stats (dict): Training statistics
    """
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Progress bar for the dataloader
            dataloader = dataloaders[phase]
            progress_bar = tqdm(dataloader, desc=f'{phase} Epoch {epoch+1}/{num_epochs}') if verbose else dataloader
            
            # Iterate over data (batch)
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar if using tqdm
                if verbose:
                    progress_bar.set_postfix({
                        'loss': loss.item(), 
                        'accuracy': torch.sum(preds == labels.data).item() / inputs.size(0)
                    })
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Store the metrics
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            if verbose:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # LR Scheduler step if it's a validation phase and scheduler is provided
            if phase == 'val' and scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                if verbose:
                    print(f'New best model found! Val accuracy: {best_acc:.4f}')
        
        if verbose:
            print()
    
    # Calculate and print training time
    time_elapsed = time.time() - since
    if verbose:
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
    
    # Store training statistics
    training_stats = {
        'training_time': f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s",
        'best_val_acc': best_acc.item(),
        'best_epoch': best_epoch
    }
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model, history, training_stats


def evaluate_model(model, test_loader, device, verbose=True):
    """
    Evaluate a trained model on a test set
    
    Args:
        model (nn.Module): Trained PyTorch model to evaluate
        test_loader (DataLoader): DataLoader for the test set
        device (torch.device): Device to evaluate on (GPU or CPU)
        verbose (bool): Whether to print progress
        
    Returns:
        accuracy (float): Classification accuracy
        all_preds (np.array): All predictions
        all_labels (np.array): All true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        progress_bar = tqdm(test_loader, desc="Evaluating") if verbose else test_loader
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    if verbose:
        print(f'Test Accuracy: {accuracy:.2f}%')
    
    return accuracy, np.array(all_preds), np.array(all_labels)
