"""
Enhanced training module with gradient clipping and early stopping.
"""
import torch
import torch.nn as nn
import time
import copy
from tqdm import tqdm
from typing import Dict, Tuple, Any, List, Optional, Union

def train_model_with_clipping(
    model: nn.Module, 
    dataloaders: Dict[str, torch.utils.data.DataLoader], 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
    num_epochs: int = 10, 
    gradient_clip_val: float = 1.0,
    early_stopping_patience: int = 15,
    early_stopping_delta: float = 0.001,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, Any]]:
    """
    Enhanced training function with gradient clipping and early stopping
    
    Args:
        model (nn.Module): PyTorch model to train
        dataloaders (dict): Dictionary of PyTorch DataLoader objects for 'train' and 'val'
        criterion: Loss function
        optimizer: Optimizer to use
        device (torch.device): Device to train on (GPU or CPU)
        scheduler: Learning rate scheduler (optional)
        num_epochs (int): Number of epochs to train for
        gradient_clip_val (float): Max norm for gradient clipping
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped
        early_stopping_delta (float): Minimum change in validation loss to qualify as improvement
        verbose (bool): Whether to print progress
        
    Returns:
        model: Best model based on validation accuracy
        history (dict): Training and validation metrics
        training_stats (dict): Training statistics
    """
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    # Early stopping parameters
    counter = 0
    best_loss = float('inf')
    
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
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                        
                        optimizer.step()
                        
                        # Step OneCycleLR per iteration
                        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                            scheduler.step()
                
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
                history['lr'].append(optimizer.param_groups[0]['lr'])
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            if verbose:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # LR Scheduler step if it's a validation phase and not OneCycleLR
            if phase == 'val':
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                elif scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                
                # Early stopping check
                if epoch_loss < best_loss - early_stopping_delta:
                    best_loss = epoch_loss
                    counter = 0
                else:
                    counter += 1
                    if verbose and counter > 0:
                        print(f"Early stopping counter: {counter}/{early_stopping_patience}")
                    
                    if counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        # Load the best model weights
                        model.load_state_dict(best_model_wts)
                        
                        # Calculate and print training time
                        time_elapsed = time.time() - since
                        if verbose:
                            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                            print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
                        
                        # Store training statistics
                        training_stats = {
                            'training_time': f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s",
                            'best_val_acc': best_acc.item(),
                            'best_epoch': best_epoch,
                            'stopped_early': True,
                            'stopped_epoch': epoch + 1
                        }
                        
                        return model, history, training_stats
            
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
        'best_epoch': best_epoch,
        'stopped_early': False
    }
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model, history, training_stats
