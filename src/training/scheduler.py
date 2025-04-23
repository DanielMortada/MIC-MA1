"""
Learning rate schedulers for model training optimization.
"""
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, OneCycleLR

def get_scheduler(scheduler_name, optimizer, **kwargs):
    """
    Factory function to create a learning rate scheduler
    
    Args:
        scheduler_name (str): Name of the scheduler ('step', 'cosine', 'plateau', 'onecycle')
        optimizer: PyTorch optimizer
        **kwargs: Additional arguments for the specific scheduler
        
    Returns:
        scheduler: PyTorch learning rate scheduler
        
    Raises:
        ValueError: If scheduler_name is invalid
    """
    if scheduler_name.lower() == 'step':
        # StepLR decreases learning rate by gamma every step_size epochs
        # Default: step_size=30, gamma=0.1
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif scheduler_name.lower() == 'cosine':
        # CosineAnnealingLR reduces learning rate following a cosine curve
        # Default: T_max=10 (cycles)
        t_max = kwargs.get('t_max', 10)
        eta_min = kwargs.get('eta_min', 0)
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        
    elif scheduler_name.lower() == 'plateau':
        # ReduceLROnPlateau reduces learning rate when a metric stops improving
        # Default: patience=5, factor=0.1
        patience = kwargs.get('patience', 5)
        factor = kwargs.get('factor', 0.1)
        min_lr = kwargs.get('min_lr', 1e-6)
        mode = kwargs.get('mode', 'min')  # 'min' for loss, 'max' for accuracy
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, 
                                patience=patience, min_lr=min_lr)
        
    elif scheduler_name.lower() == 'onecycle':
        # OneCycleLR implements the One Cycle Policy
        # Required: max_lr, steps_per_epoch, epochs
        max_lr = kwargs.get('max_lr')
        steps_per_epoch = kwargs.get('steps_per_epoch')
        epochs = kwargs.get('epochs')
        pct_start = kwargs.get('pct_start', 0.3)
        
        if not all([max_lr, steps_per_epoch, epochs]):
            raise ValueError("OneCycleLR requires max_lr, steps_per_epoch, and epochs")
            
        return OneCycleLR(optimizer, max_lr=max_lr, 
                          steps_per_epoch=steps_per_epoch, 
                          epochs=epochs,
                          pct_start=pct_start)
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")

def warmup_cosine_schedule(optimizer, warmup_steps, total_steps):
    """
    Custom learning rate scheduler with linear warmup followed by cosine decay
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps (int): Number of warmup steps
        total_steps (int): Total number of training steps
        
    Returns:
        function: Learning rate scheduler function to call on each step
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
