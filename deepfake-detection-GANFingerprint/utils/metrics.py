"""
Metrics utilities for model evaluation.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve,
    average_precision_score
)


def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Prediction probabilities
        threshold: Classification threshold
    
    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_probs),
        'ap': average_precision_score(y_true, y_pred_probs)
    }
    
    return metrics


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        array: Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_roc_curve_data(y_true, y_pred_probs):
    """
    Get ROC curve data points.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Prediction probabilities
    
    Returns:
        tuple: (fpr, tpr, thresholds)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    return fpr, tpr, thresholds


def get_precision_recall_curve_data(y_true, y_pred_probs):
    """
    Get precision-recall curve data points.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Prediction probabilities
    
    Returns:
        tuple: (precision, recall, thresholds)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
    return precision, recall, thresholds


def find_optimal_threshold(y_true, y_pred_probs, metric='f1'):
    """
    Find the optimal threshold based on a specific metric.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Prediction probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
    
    Returns:
        float: Optimal threshold
    """
    best_threshold = 0.5
    best_score = 0.0
    
    # Try different thresholds
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_probs >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Metric {metric} not supported")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold