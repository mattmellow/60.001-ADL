"""
Training script for the GANFingerprint deepfake detection model.
"""
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import config
from data_loader import get_dataloaders, get_dataset_stats
from models import FingerprintNet
from utils.metrics import compute_metrics
from utils.reproducibility import set_all_seeds, get_random_state
from utils.experiment import ExperimentTracker


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Returns a learning rate scheduler with warmup and cosine annealing.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    """
    Training function.
    """
    #Set seeds at beginning of training
    set_all_seeds(config.SEED)


    # Create directories if they don't exist
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    # os.makedirs(config.EXPERIMENT_LOGS, exist_ok=True)
 
    # Get data loaders
    train_loader, val_loader, _ = get_dataloaders(seed=config.SEED)
    
    config_log = {
        'BACKBONE': config.BACKBONE,
        'EMBEDDING_DIM': config.EMBEDDING_DIM,
        'DROPOUT_RATE': config.DROPOUT_RATE,
        'BATCH_SIZE': config.BATCH_SIZE,
        'LEARNING_RATE': config.LEARNING_RATE,
        'SEED': config.SEED,
        'NUM_EPOCHS': config.NUM_EPOCHS,
        'DEVICE': str(config.DEVICE),  # Convert torch.device to string
        'INPUT_SIZE': config.INPUT_SIZE,
        'USE_AMP': config.USE_AMP,
    }

    # Create experiment tracker
    tracker = ExperimentTracker(config.EXPERIMENT_NAME, base_dir=config.EXPERIMENT_LOGS)
    tracker.log("Starting training with configuration:")
    tracker.save_config(config_log)
    tracker.log(f"Using device: {config.DEVICE}")

    # Print dataset stats
    get_dataset_stats()
    tracker.log(f"Dataset loaded successfully")
    
    # Initialize model
    model = FingerprintNet(backbone=config.BACKBONE)
    model = model.to(config.DEVICE)
    tracker.log(f"Model initialized with backbone: {config.BACKBONE}")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=config.NUM_EPOCHS
    )

    
    # Gradient scaler for mixed precision training
    scaler = GradScaler('cuda') if config.USE_AMP else None
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(config.LOG_DIR, config.EXPERIMENT_NAME))
    
    # Variables for early stopping
    best_val_auc = 0
    patience_counter = 0    
    start_epoch = 0
    best_combined_score = 0.0
    
    metric_weights = {
        'accuracy': 0.10,
        'precision': 0.05, 
        'recall': 0.70,      # Higher weight for recall
        'f1': 0.10,
        'auc': 0.05
    }

    # Check if resuming from checkpoint
    if args.resume_checkpoint:
        tracker.log(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=config.DEVICE, weights_only=False)
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                tracker.log("Resumed scheduler state successfully")
            except Exception as e:
                tracker.log(f"Warning: Could not load scheduler state: {e}")
                tracker.log("Creating new scheduler from scratch")
        
        # Set training state
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        best_val_auc = checkpoint.get('val_auc', 0.0)
        
        tracker.log(f"Resuming from epoch {start_epoch}, best val AUC so far: {best_val_auc:.4f}")


    # Training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        for images, labels in train_progress:
            images = images.to(config.DEVICE)
            labels = labels.float().to(config.DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            if config.USE_AMP:
                # Mixed precision training
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate training metrics
        train_metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]")
            for images, labels in val_progress:
                images = images.to(config.DEVICE)
                labels = labels.float().to(config.DEVICE)
                
                if config.USE_AMP:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item()
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate validation metrics
        val_metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        val_loss /= len(val_loader)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('AUC/train', train_metrics['auc'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        writer.add_scalar('Recall/train', train_metrics['recall'], epoch)
        writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
        writer.add_scalar('F1-Score/train', train_metrics['f1'], epoch)
        writer.add_scalar('F1-Score/val', val_metrics['f1'], epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}, Recall: {train_metrics['recall']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        # Calculate validation metrics, weighted between AUC and Recall
        val_metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        
        # Calculate the weighted combination score
        combined_score = sum(metric_weights[metric] * val_metrics[metric] 
                         for metric in metric_weights.keys())
        
        # Log epoch results with experiment tracker
        epoch_time = time.time() - epoch_start_time
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'train_metrics': {k: float(v) for k, v in train_metrics.items()},
            'val_metrics': {k: float(v) for k, v in val_metrics.items()},
            'combined_score': float(combined_score),
            'learning_rate': float(scheduler.get_last_lr()[0]),
            'epoch_time_seconds': float(epoch_time)
        }
        tracker.save_results(epoch_results)
        
        # Log summary to experiment tracker
        tracker.log(f"Epoch {epoch+1}/{config.NUM_EPOCHS} completed in {epoch_time:.2f}s")
        tracker.log(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        tracker.log(f"Combined Score: {combined_score:.4f} (Weights: Accuracy={metric_weights['accuracy']}, "
            f"Precision={metric_weights['precision']}, Recall={metric_weights['recall']}, "
            f"F1={metric_weights['f1']}, AUC={metric_weights['auc']})")

        config_dict = {
            'BACKBONE': config.BACKBONE,
            'EMBEDDING_DIM': config.EMBEDDING_DIM,
            'DROPOUT_RATE': config.DROPOUT_RATE,
            'BATCH_SIZE': config.BATCH_SIZE,
            'LEARNING_RATE': config.LEARNING_RATE,
            'SEED': config.SEED,
            # Add other parameters you need to restore
        }

        # Check if this is the best model according to the combined score
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            patience_counter = 0
            
            # Save the best model
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{config.EXPERIMENT_NAME}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'combined_score': best_combined_score,
                'val_metrics': val_metrics,
                'metric_weights': metric_weights,  # Save the weights used
                'random_state': get_random_state(),
                'config': config_dict
            }, checkpoint_path)
            tracker.log(f"New best model saved (Combined Score: {best_combined_score:.4f})")
        else:
            patience_counter += 1
            tracker.log(f"No improvement in combined score. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

        # Save regular checkpoint
        if (epoch + 1) % 5 == 0 or epoch == config.NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{config.EXPERIMENT_NAME}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
                'val_auc': val_metrics['auc'],
                'val_metrics': val_metrics,
                'random_state': get_random_state(),
                # Don't save config in regular checkpoints to avoid issues
            }, checkpoint_path)
            tracker.log(f"Saved checkpoint to {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            tracker.log(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    tracker.log(f"Training completed. Best combined score: {best_combined_score:.4f}")

    # Save final summary
    final_results = {
        'best_combined_score': float(best_combined_score),
        'total_epochs_trained': epoch + 1,
        'early_stopped': patience_counter >= config.EARLY_STOPPING_PATIENCE,
        'final_learning_rate': float(scheduler.get_last_lr()[0]),
        'metric_weights': metric_weights
    }
    tracker.save_results(final_results)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GANFingerprint model for deepfake detection')
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT, help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--backbone', type=str, default=config.BACKBONE, help='Backbone model')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')

    args = parser.parse_args()
    
    # Override config values with command line arguments
    config.DATA_ROOT = args.data_root
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.NUM_EPOCHS = args.epochs
    config.BACKBONE = args.backbone
    config.USE_AMP = not args.no_amp
    
    # Run training
    train(args)