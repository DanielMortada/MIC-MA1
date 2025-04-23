"""
Metrics for evaluating musical instrument classification models.
"""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def compute_metrics(true_labels, predictions, class_names=None):
    """
    Compute classification metrics from predictions
    
    Args:
        true_labels (np.array): Ground truth labels
        predictions (np.array): Predicted labels
        class_names (list): List of class names (optional)
        
    Returns:
        metrics (dict): Dictionary of metrics
    """
    # Calculate overall accuracy
    acc = accuracy_score(true_labels, predictions)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None
    )
    
    # Calculate macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    # Create metrics dictionary
    metrics = {
        'accuracy': acc,
        'class_metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'weighted_avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        }
    }
    
    # Add class names if provided
    if class_names is not None:
        metrics['class_names'] = class_names
    
    return metrics

def get_confusion_matrix(true_labels, predictions, normalize=None):
    """
    Compute confusion matrix for classification results
    
    Args:
        true_labels (np.array): Ground truth labels
        predictions (np.array): Predicted labels
        normalize (str): Normalization option ('true', 'pred', 'all', or None)
        
    Returns:
        cm (np.array): Confusion matrix
    """
    return confusion_matrix(true_labels, predictions, normalize=normalize)
