"""
Visualization utilities for model analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import average_precision_score
# roc_curve, precision_recall_curve
import torch
# from torch.utils.data import DataLoader
# import torchvision

from utils.metrics import get_roc_curve_data, get_precision_recall_curve_data


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='accuracy'):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: List of training metrics
        val_metrics: List of validation metrics
        metric_name: Name of the metric to plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    ax2.plot(train_metrics, label=f'Train {metric_name.capitalize()}')
    ax2.plot(val_metrics, label=f'Validation {metric_name.capitalize()}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_name.capitalize())
    ax2.set_title(f'Training and Validation {metric_name.capitalize()}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Color map
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt


def plot_roc_curve(y_true, y_pred_probs):
    """
    Plot ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Prediction probabilities
    """
    fpr, tpr, _ = get_roc_curve_data(y_true, y_pred_probs)
    roc_auc = np.trapz(tpr, fpr)  # Area under the curve
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    return plt


def plot_precision_recall_curve(y_true, y_pred_probs):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Prediction probabilities
    """
    precision, recall, _ = get_precision_recall_curve_data(y_true, y_pred_probs)
    ap = average_precision_score(y_true, y_pred_probs)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    return plt


def visualize_model_predictions(model, dataloader, device, num_images=8):
    """
    Visualize model predictions on a batch of images.
    
    Args:
        model: Trained model
        dataloader: DataLoader for images
        device: Device to run inference on
        num_images: Number of images to visualize
    """
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    labels = labels[:num_images].numpy()
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    
    # Plot images with predictions
    fig = plt.figure(figsize=(12, 8))
    for i in range(num_images):
        ax = plt.subplot(2, num_images//2, i + 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        title = f"True: {'Real' if labels[i] == 1 else 'Fake'}\n"
        title += f"Pred: {'Real' if preds[i] == 1 else 'Fake'} ({probs[i]:.2f})"
        ax.set_title(title, color='green' if preds[i] == labels[i] else 'red')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_attention_maps(model, image_tensor, device):
    """
    Visualize attention maps for an image.
    
    Args:
        model: Model with attention mechanism
        image_tensor: Input image tensor
        device: Device to run inference on
    """
    # Ensure image is on the correct device
    image_tensor = image_tensor.to(device)
    
    # Get model features and attention maps
    model.eval()
    with torch.no_grad():
        # This requires model modification to return attention maps
        # For demonstration purposes only
        outputs, attention_maps = model.get_attention_maps(image_tensor)
    
    # Convert image for visualization
    img = image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    # Plot image and attention maps
    num_maps = min(4, len(attention_maps))
    fig, axes = plt.subplots(1, num_maps + 1, figsize=(15, 4))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for i in range(num_maps):
        attn_map = attention_maps[i].cpu().squeeze().numpy()
        axes[i+1].imshow(attn_map, cmap='jet')
        axes[i+1].set_title(f'Attention Map {i+1}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    return fig