# Implementation Plan for Improved Deeper CNN Model

This document outlines a structured approach to improve the performance of the Deeper CNN model, which achieved 86.67% test accuracy in our model comparison study. Our objective is to close the gap with the ResNet-18 baseline (99.33% test accuracy).

## Current Configuration

Based on our notebook implementation in 6_Deeper_CNN_Optimisation.ipynb, the optimized model uses:

- **Architecture**:
  - Residual connections (enabled)
  - Selective attention mechanisms (enabled for deeper layers i≥3 only)
  - Convolutional layers: [32, 64, 128, 256, 512, 640] (increased width in final layer)
  - Fully connected layers: [512, 256]
- **Regularization**:
  - Graduated dropout rates: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3]
  - Batch normalization (enabled)
- **Training**:
  - AdamW optimizer with weight decay: 0.0005
  - Learning rate: 0.0003 with max_lr of 0.001 using OneCycleLR
  - Early stopping patience: 25 epochs with minimum delta: 0.001
  - Gradient clipping: 2.0
  - Mixed precision training: enabled when CUDA is available
- **Data**:
  - Augmentation strength: medium
  - Class-specific augmentations for challenging classes

## Recommended Implementations

The following implementations should be applied in order of priority:

### 1. Architecture Adjustments

#### 1.1. Reintroduce Selective Attention Mechanisms

```python
# Re-enable attention but only in deeper layers
config['model']['use_attention'] = True 

# Modify the EnhancedFlexibleCNN _make_feature_extractor method to add attention only in specific layers
# Inside src/models/enhanced_cnn.py, modify the attention insertion condition:
if self.use_attention and i >= len(self.conv_layers) - 2:  # Only in the last two layers
    layers.append(ChannelAttention(filters))
```

#### 1.2. Residual Connection Refinement

```python
# Modify ResidualBlock to include optional bottleneck design
class ResidualBlock(nn.Module):
    """Enhanced residual block with bottleneck option."""
    def __init__(self, in_channels, out_channels, stride=1, use_bottleneck=False):
        super(ResidualBlock, self).__init__()
        
        self.use_bottleneck = use_bottleneck
        
        if use_bottleneck:
            # Bottleneck design: reduce channels, then 3x3 conv, then expand channels
            bottleneck_channels = out_channels // 4
            self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(bottleneck_channels)
            self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, 
                                  stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(bottleneck_channels)
            self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            # Standard residual block
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                  stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                  stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        if self.use_bottleneck:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

# Enable bottleneck in deeper layers
config['model']['use_bottleneck'] = True
```

#### 1.3. Adjust Network Width in Higher Layers

```python
# Modify convolutional layer configuration to be wider in higher layers
config['model']['conv_layers'] = [32, 64, 128, 256, 512, 640]  # Increased last layer width
```

### 2. Training Strategy Optimization

#### 2.1. Extended Training Duration

```python
# Increase maximum epochs
config['training']['num_epochs'] = 125  # Increased from 75
```

#### 2.2. Learning Rate Adjustment

```python
# Increase max learning rate slightly
max_lr = 0.005  # Increased from 0.003

# Adjust learning rate schedule
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    pct_start=0.4,  # Extended warmup period (from 0.3)
    div_factor=10.0,
    final_div_factor=1000.0
)
```

#### 2.3. Mixed Precision Training

```python
# Enable mixed precision training for faster computation and larger batch sizes
from torch.cuda.amp import GradScaler, autocast

# Initialize the scaler
scaler = GradScaler()

# In the training loop:
for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    # Backward and optimize with scaling
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
    scaler.step(optimizer)
    scaler.update()
    
    # Step scheduler
    scheduler.step()
```

### 3. Data Augmentation Enhancements

#### 3.1. Return to Strong Augmentation

```python
# Update augmentation strength
augmentation_strength = 'strong'  # Changed from 'medium'

# Get transforms with stronger augmentation
train_transform, val_transform = AdvancedAugmentation.get_advanced_transforms(
    img_size=224,
    augmentation_strength=augmentation_strength
)
```

#### 3.2. Class-Specific Augmentation

```python
# Implement class-specific augmentation for challenging classes
from torchvision.transforms import v2

class ClassSpecificTransform:
    def __init__(self, base_transform, special_classes, special_transform):
        self.base_transform = base_transform
        self.special_classes = special_classes
        self.special_transform = special_transform
    
    def __call__(self, img, class_name):
        if class_name in self.special_classes:
            return self.special_transform(img)
        return self.base_transform(img)

# Create stronger transform for challenging instrument classes
challenging_classes = ['Didgeridoo', 'Flute', 'Trombone', 'Sitar', 'Alphorn', 
                      'Castanets', 'Clarinet', 'Piano', 'Steel drum']

stronger_transform = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.2),  # More aggressive vertical flipping
    v2.RandomRotation(degrees=30),  # More rotation
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # More color jitter
    v2.RandomErasing(p=0.4, scale=(0.02, 0.2)),  # More aggressive erasing
    v2.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # Add affine transforms
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use class-specific transforms in the dataset
class_specific_transform = ClassSpecificTransform(
    base_transform=train_transform,
    special_classes=challenging_classes,
    special_transform=stronger_transform
)

# Modify the dataset class to use class-specific augmentation
class EnhancedInstrumentDataset(InstrumentDataset):
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        class_name = self.classes[label]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img, class_name)
        
        return img, label
```

#### 3.3. Test Time Augmentation

```python
# Implement test-time augmentation (TTA) for evaluation
def tta_evaluation(model, test_loader, device, num_augmentations=5):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_size = inputs.size(0)
            labels = labels.to(device)
            all_labels.extend(labels.cpu().numpy())
            
            # Initialize predictions for this batch
            batch_predictions = torch.zeros(batch_size, model.num_classes).to(device)
            
            # Original prediction
            outputs = model(inputs.to(device))
            batch_predictions += outputs
            
            # Augmented predictions
            for _ in range(num_augmentations - 1):
                # Apply test-time augmentations
                augmented_inputs = test_time_transforms(inputs)
                outputs = model(augmented_inputs.to(device))
                batch_predictions += outputs
            
            # Average predictions
            batch_predictions /= num_augmentations
            
            # Get predicted class
            _, preds = torch.max(batch_predictions, 1)
            all_predictions.extend(preds.cpu().numpy())
    
    accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    return accuracy, all_predictions, all_labels

# Test-time transforms (moderate augmentations)
test_time_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=10),
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 4. Regularization Adjustments

#### 4.1. Refined Dropout Strategy

```python
# Try a more aggressive yet gradual dropout pattern
config['model']['dropout'] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # More gradual start
```

#### 4.2. Weight Decay Optimization by Layer Type

```python
# Implement layer-wise weight decay with PyTorch
from torch.optim import AdamW

# Group parameters by layer type with different weight decay
def get_layer_wise_parameters(model):
    conv_params = []
    bn_params = []
    fc_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'conv' in name:
                conv_params.append(param)
            elif 'bn' in name:
                bn_params.append(param)
            elif 'classifier' in name or 'fc' in name:
                fc_params.append(param)
    
    return [
        {'params': conv_params, 'weight_decay': 0.0003},  # Lower weight decay for conv layers
        {'params': bn_params, 'weight_decay': 0.0},       # No weight decay for batch norm
        {'params': fc_params, 'weight_decay': 0.001}      # Higher weight decay for FC layers
    ]

# Create optimizer with layerwise weight decay
optimizer = AdamW(
    get_layer_wise_parameters(model),
    lr=lr,
    betas=(beta1, beta2)
)
```

#### 4.3. Focal Loss for Class Imbalance

```python
# Implement Focal Loss for better handling of hard examples
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:  # 'none'
            return F_loss

# Use focal loss for training
criterion = FocalLoss(gamma=2.0)
```

### 5. Ensemble Approaches

#### 5.1. Stochastic Weight Averaging (SWA)

```python
# Implement Stochastic Weight Averaging for better generalization
from torch.optim.swa_utils import AveragedModel, SWALR

# Create SWA model
swa_model = AveragedModel(model)

# SWA scheduler (constant LR)
swa_scheduler = SWALR(optimizer, swa_lr=0.001)

# Start SWA after 75% of training
swa_start = int(0.75 * num_epochs)

# During training:
for epoch in range(num_epochs):
    # Regular training for first 75% of epochs
    if epoch < swa_start:
        # Regular training with OneCycle scheduling
        train_one_epoch(model, ...)
        scheduler.step()
    else:
        # After swa_start, switch to SWA
        train_one_epoch(model, ...)
        swa_model.update_parameters(model)
        swa_scheduler.step()
    
    # Validation
    validate_one_epoch(model, ...)

# Update batch normalization for SWA model
torch.optim.swa_utils.update_bn(train_loader, swa_model)

# Use SWA model for testing
validate_one_epoch(swa_model, ...)
```

#### 5.2. Simple Model Averaging

```python
# Train multiple models with different random seeds and average predictions
def train_multiple_models(config, seeds, dataloaders, num_models=3):
    models = []
    
    for i, seed in enumerate(seeds):
        set_seed(seed)
        print(f"\nTraining model {i+1}/{num_models} with seed {seed}")
        
        # Create and train model
        model = create_enhanced_flexible_cnn(config)
        model = model.to(device)
        
        # Train with the same configuration
        trained_model, _, _ = train_model_with_clipping(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=num_epochs,
            gradient_clip_val=gradient_clip_val,
            early_stopping_patience=early_stopping_patience,
            early_stopping_delta=early_stopping_delta,
            verbose=True
        )
        
        models.append(trained_model)
    
    return models

# Evaluate ensemble by averaging predictions
def evaluate_ensemble(models, test_loader, device):
    for model in models:
        model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            all_labels.extend(labels.cpu().numpy())
            
            # Initialize predictions tensor
            ensemble_preds = torch.zeros(batch_size, models[0].num_classes).to(device)
            
            # Add predictions from each model
            for model in models:
                outputs = model(inputs)
                ensemble_preds += outputs
            
            # Average predictions
            ensemble_preds /= len(models)
            
            # Get predicted class
            _, preds = torch.max(ensemble_preds, 1)
            all_predictions.extend(preds.cpu().numpy())
    
    accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    return accuracy, all_predictions, all_labels

# Use the ensemble approach
seed_list = [42, 123, 456]
ensemble_models = train_multiple_models(config, seed_list, dataloaders, num_models=3)
ensemble_acc, ensemble_preds, ensemble_labels = evaluate_ensemble(ensemble_models, test_loader, device)
print(f"Ensemble accuracy: {ensemble_acc:.2f}%")
```

## Implementation Guidelines

1. Start with architecture adjustments (Section 1) since they require minimal changes to the existing code structure.
2. Next, implement the data augmentation enhancements (Section 3) which can significantly improve the model's generalization.
3. Apply the regularization adjustments (Section 4) to fine-tune the training process.
4. Finally, if the model still underperforms, implement the ensemble approaches (Section 5).

For each implementation, evaluate the model on the test set and track the accuracy improvement. Compare the results to both the current optimized model (81%) and the original Deeper CNN (86.67%).

## Expected Outcomes

By implementing these changes, we expect to achieve a test accuracy of at least 87-88%, exceeding the original Deeper CNN model. The most significant improvements will likely come from:

1. Selective attention mechanisms (Section 1.1)
2. Extended training duration (Section 2.1)
3. Return to strong augmentation (Section 3.1)
4. Ensemble approaches (Section 5)

If time constraints prevent implementing all the recommendations, prioritize the architecture adjustments and data augmentation enhancements as they typically provide the most immediate benefits.
