"""
Evaluation script for the trained GANFingerprint model.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from torch.amp import autocast

import config
from data_loader import get_dataloaders
from models import FingerprintNet
from utils.metrics import compute_metrics, compute_confusion_matrix
from utils.visualization import plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
from utils.reproducibility import set_all_seeds, set_random_state

# import torch.serialization
# # Allow numpy scalar type to be loaded
# torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])

def evaluate(checkpoint_path, output_dir):
    """
    Evaluate the model on the test set and generate performance visualizations.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        output_dir (str): Directory to save evaluation results
    """
    # Set base seeds for reproducibility
    set_all_seeds(config.SEED)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test data loader with the same seed used in training
    _, _, test_loader = get_dataloaders(seed=config.SEED)
    
    # Initialize model
    model = FingerprintNet(backbone=config.BACKBONE)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optionally restore the exact random state from training
    if 'random_state' in checkpoint:
        try:
            set_random_state(checkpoint['random_state'])
            print("Restored random state from checkpoint")
        except Exception as e:
            print("Could not restore random state - using default seed")
            #R eset with default seed 
            set_all_seeds(config.SEED)
    
    model = model.to(config.DEVICE)
    model.eval()
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_prob_preds = []
    
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Evaluating")
        for images, labels in test_progress:
            images = images.to(config.DEVICE)
            
            # Forward pass with mixed precision if enabled
            if config.USE_AMP:
                with autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Get predictions
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            # Store results
            all_prob_preds.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

     
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_prob_preds = np.array(all_prob_preds)
    
    # Calculate metrics
    metrics = compute_metrics(all_labels, all_prob_preds)
    print("Test Set Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"AUC-ROC: {metrics['auc']:.4f}")
    
    # Save metrics to a text file
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC: {metrics['auc']:.4f}\n")
    
    # Calculate and plot confusion matrix
    cm = compute_confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm, classes=['Fake', 'Real'], normalize=False)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plot_roc_curve(all_labels, all_prob_preds)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plot_precision_recall_curve(all_labels, all_prob_preds)
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=300, bbox_inches='tight')
    
    print(f"Evaluation results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GANFingerprint model for deepfake detection')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    