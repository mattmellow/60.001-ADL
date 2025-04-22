"""
Inference script for applying the GANFingerprint model on new images with Grad-CAM visualization.
"""
import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.amp import autocast
import torchvision.transforms as transforms

import config
from models import FingerprintNet
from utils.reproducibility import set_all_seeds
from utils.gradcam import visualize_gradcam


def predict_image_calibrated(model, image_path, transform):
    """
    Make a prediction on a single image with calibrated probability.
    
    Args:
        model: The FingerprintNet model
        image_path: Path to the image file
        transform: Image transformation pipeline
        
    Returns:
        Tuple of (calibrated probability, predicted class)
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(image_tensor)
    
    # Get raw logit 
    raw_logit = output.item()
    
    # Original probability
    orig_prob = torch.sigmoid(output).item()
    
    # Apply calibration for fake images
    if orig_prob < 0.5:  # Predicted as fake
        # Map [0, 0.5] range to [1.0, 0] - this inverts the scale for fake images
        calibrated_prob = 1.0 - (2.0 * orig_prob)
    else:  # Predicted as real
        calibrated_prob = orig_prob
    
    pred_class = "Real" if orig_prob >= 0.5 else "Fake"
    
    print(f"Raw logit: {raw_logit:.6f}, Original prob: {orig_prob:.6f}, Calibrated: {calibrated_prob:.6f}")
    
    return calibrated_prob, pred_class


def extract_true_label(filename):
    """
    Extract the true label from the image filename.
    Assumes filenames contain 'real' or 'fake' to indicate the true class.
    
    Args:
        filename: The filename to extract the label from
        
    Returns:
        "Real" or "Fake" based on the filename, or "Unknown" if can't determine
    """
    basename = os.path.basename(filename).lower()
    
    if 'real' in basename:
        return "Real"
    elif 'fake' in basename:
        return "Fake"
    else:
        return "Unknown"


def calculate_metrics(true_labels, pred_labels):
    """
    Calculate accuracy, precision, recall and F1 score.
    
    Args:
        true_labels: List of true labels ("Real" or "Fake")
        pred_labels: List of predicted labels ("Real" or "Fake")
        
    Returns:
        Dictionary with accuracy, precision, recall and F1 metrics
    """
    # Count total predictions
    total = len(true_labels)
    
    # Count correct predictions
    correct = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    
    # For "Real" class metrics
    true_positives = sum(1 for true, pred in zip(true_labels, pred_labels) 
                        if true == "Real" and pred == "Real")
    false_positives = sum(1 for true, pred in zip(true_labels, pred_labels) 
                         if true == "Fake" and pred == "Real")
    false_negatives = sum(1 for true, pred in zip(true_labels, pred_labels) 
                         if true == "Real" and pred == "Fake")
    true_negatives = sum(1 for true, pred in zip(true_labels, pred_labels) 
                        if true == "Fake" and pred == "Fake")
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    # Precision for "Real" class
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Recall for "Real" class
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision, 
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives
    }


def visualize_result(image_path, prob, pred_class, true_class=None, output_path=None):
    """
    Visualize the prediction result.
    
    Args:
        image_path: Path to the image
        prob: Prediction probability
        pred_class: Predicted class
        true_class: True class (optional)
        output_path: Path to save the visualization (optional)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Display image
    plt.imshow(image)
    plt.axis('off')
    
    # Add prediction text
    color = 'green' if pred_class == 'Real' else 'red'
    
    # Different title based on whether we have the true class
    if true_class and true_class != "Unknown":
        match_status = "✓" if pred_class == true_class else "✗"
        plt.title(f"Pred: {pred_class} ({prob:.4f}) | True: {true_class} {match_status}", 
                 color='green' if pred_class == true_class else 'red', 
                 fontsize=14)
    else:
        plt.title(f"Prediction: {pred_class} ({prob:.4f})", color=color, fontsize=16)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def run_inference(checkpoint_path, input_path, output_dir=None, batch_mode=False, use_gradcam=False):
    """
    Run inference on one or multiple images.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        input_path: Path to an image or directory of images
        output_dir: Directory to save results
        batch_mode: Whether to process a directory of images
        use_gradcam: Whether to generate Grad-CAM visualizations
    """
    # Set seeds for reproducibility
    set_all_seeds(config.SEED)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = FingerprintNet(backbone=config.BACKBONE)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    
    # Image transforms (same as validation/test)
    transform = transforms.Compose([
        transforms.Resize(config.INPUT_SIZE),
        transforms.CenterCrop(config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Single image mode
    if not batch_mode:
        true_class = extract_true_label(input_path)
        
        if use_gradcam:
            # Load and preprocess the image for Grad-CAM
            orig_image = Image.open(input_path).convert('RGB')
            image_tensor = transform(orig_image).to(config.DEVICE)
            
            # Get target layer for Grad-CAM
            from utils.gradcam import get_gradcam_layer, GradCAM, generate_gradcam
            target_layer = get_gradcam_layer(model)
            
            # Generate Grad-CAM visualization
            raw_logit, heatmap, superimposed = generate_gradcam(model, image_tensor, orig_image)
            
            # Calculate prediction
            orig_prob = torch.sigmoid(torch.tensor(raw_logit)).item()
            if orig_prob < 0.5:  # Predicted as fake
                calibrated_prob = 1.0 - (2.0 * orig_prob)
            else:  # Predicted as real
                calibrated_prob = orig_prob
            
            pred_class = "Real" if orig_prob >= 0.5 else "Fake"
            
            # Create and save visualization
            if output_dir:
                output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_gradcam.png")
                
                # Create visualization with improved layout
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                fig.subplots_adjust(top=0.85)  # Make more room for title
                
                # Original image
                ax1.imshow(orig_image)
                ax1.set_title("Original Image", fontsize=14, pad=10)
                ax1.axis('off')
                
                # Grad-CAM visualization
                ax2.imshow(superimposed)
                
                # Set title color based on prediction
                title_color = 'green' if pred_class == 'Real' else 'red'
                ax2.set_title(f"Grad-CAM: {pred_class} ({calibrated_prob:.4f})", 
                            fontsize=14, color=title_color, pad=10)
                ax2.axis('off')
                
                # Add information about prediction above the figures
                if true_class and true_class != "Unknown":
                    correct = pred_class == true_class
                    result_color = 'green' if correct else 'red'
                    
                    # Create main title with appropriate color and check/x mark
                    check_mark = "✓" if correct else "✗"
                    title = f"Prediction: {pred_class} ({calibrated_prob:.4f})\nTrue Class: {true_class} ({check_mark})"
                    plt.suptitle(title, fontsize=16, y=0.98, color=result_color)
                else:
                    plt.suptitle(f"Prediction: {pred_class} ({calibrated_prob:.4f})", 
                               fontsize=16, y=0.98, color=title_color)
                
                # Add extra space between plots
                plt.tight_layout(pad=3.0)
                
                # Save visualization
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
            else:
                # Display visualization with improved layout
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                fig.subplots_adjust(top=0.85)  # Make more room for title
                
                # Original image
                ax1.imshow(orig_image)
                ax1.set_title("Original Image", fontsize=14, pad=10)
                ax1.axis('off')
                
                # Grad-CAM visualization
                ax2.imshow(superimposed)
                
                # Set title color based on prediction
                title_color = 'green' if pred_class == 'Real' else 'red'
                ax2.set_title(f"Grad-CAM: {pred_class} ({calibrated_prob:.4f})", 
                            fontsize=14, color=title_color, pad=10)
                ax2.axis('off')
                
                # Add information about prediction above the figures
                if true_class and true_class != "Unknown":
                    correct = pred_class == true_class
                    result_color = 'green' if correct else 'red'
                    
                    # Create main title with appropriate color and check/x mark
                    check_mark = "✓" if correct else "✗"
                    title = f"Prediction: {pred_class} ({calibrated_prob:.4f})\nTrue Class: {true_class} ({check_mark})"
                    plt.suptitle(title, fontsize=16, y=0.98, color=result_color)
                else:
                    plt.suptitle(f"Prediction: {pred_class} ({calibrated_prob:.4f})", 
                               fontsize=16, y=0.98, color=title_color)
                
                # Add extra space between plots
                plt.tight_layout(pad=3.0)
                plt.show()
                
            prob = calibrated_prob
        else:
            # Use standard visualization
            prob, pred_class = predict_image_calibrated(model, input_path, transform)
            
            if output_dir:
                output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_pred.png")
                visualize_result(input_path, prob, pred_class, true_class, output_path)
            else:
                visualize_result(input_path, prob, pred_class, true_class)
        
        print(f"Prediction for {os.path.basename(input_path)}: {pred_class} (Confidence: {prob:.4f})")
        if true_class != "Unknown":
            correct = pred_class == true_class
            print(f"True class: {true_class} | Prediction {'correct' if correct else 'incorrect'}")
    
    # Batch mode (directory of images)
    else:
        # Create a results folder named after the input folder
        input_folder_name = os.path.basename(os.path.normpath(input_path))
        if output_dir:
            # Create a folder named input_folder_name + "_results"
            results_folder_name = f"{input_folder_name}_results"
            batch_output_dir = os.path.join(output_dir, results_folder_name)
            os.makedirs(batch_output_dir, exist_ok=True)
            
            # Create a subfolder for GradCAM visualizations if needed
            if use_gradcam:
                gradcam_dir = os.path.join(batch_output_dir, "gradcam")
                os.makedirs(gradcam_dir, exist_ok=True)
        else:
            batch_output_dir = None
            gradcam_dir = None
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
        
        # Process each image
        results = []
        for img_path in tqdm(image_files, desc=f"Processing images from {input_folder_name}"):
            true_class = extract_true_label(img_path)
            
            if use_gradcam:
                # Load and preprocess the image for Grad-CAM
                orig_image = Image.open(img_path).convert('RGB')
                image_tensor = transform(orig_image).to(config.DEVICE)
                
                # Get target layer for Grad-CAM
                from utils.gradcam import get_gradcam_layer, GradCAM, generate_gradcam
                target_layer = get_gradcam_layer(model)
                
                # Generate Grad-CAM visualization
                raw_logit, heatmap, superimposed = generate_gradcam(model, image_tensor, orig_image)
                
                # Calculate prediction
                orig_prob = torch.sigmoid(torch.tensor(raw_logit)).item()
                if orig_prob < 0.5:  # Predicted as fake
                    calibrated_prob = 1.0 - (2.0 * orig_prob)
                else:  # Predicted as real
                    calibrated_prob = orig_prob
                
                pred_class = "Real" if orig_prob >= 0.5 else "Fake"
                
                # Create and save visualization with improved layout
                if batch_output_dir:
                    output_path = os.path.join(
                        gradcam_dir, 
                        f"{os.path.splitext(os.path.basename(img_path))[0]}_gradcam.png"
                    )
                    
                    # Create visualization with improved layout
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    fig.subplots_adjust(top=0.85)  # Make more room for title
                    
                    # Original image
                    ax1.imshow(orig_image)
                    ax1.set_title("Original Image", fontsize=14, pad=10)
                    ax1.axis('off')
                    
                    # Grad-CAM visualization
                    ax2.imshow(superimposed)
                    
                    # Set title color based on prediction
                    title_color = 'green' if pred_class == 'Real' else 'red'
                    ax2.set_title(f"Grad-CAM: {pred_class} ({calibrated_prob:.4f})", 
                                fontsize=14, color=title_color, pad=10)
                    ax2.axis('off')
                    
                    # Add information about prediction above the figures
                    if true_class and true_class != "Unknown":
                        correct = pred_class == true_class
                        result_color = 'green' if correct else 'red'
                        
                        # Create main title with appropriate color and check/x mark
                        check_mark = "✓" if correct else "✗"
                        title = f"Prediction: {pred_class} ({calibrated_prob:.4f})\nTrue Class: {true_class} ({check_mark})"
                        plt.suptitle(title, fontsize=16, y=0.98, color=result_color)
                    else:
                        plt.suptitle(f"Prediction: {pred_class} ({calibrated_prob:.4f})", 
                                   fontsize=16, y=0.98, color=title_color)
                    
                    # Add extra space between plots
                    plt.tight_layout(pad=3.0)
                    
                    # Save visualization
                    plt.savefig(output_path, bbox_inches='tight', dpi=150)
                    plt.close()
                
                prob = calibrated_prob
            else:
                # Use standard inference
                prob, pred_class = predict_image_calibrated(model, img_path, transform)
                
                # Save visualization if output directory is specified
                if batch_output_dir:
                    output_path = os.path.join(
                        batch_output_dir, 
                        f"{os.path.splitext(os.path.basename(img_path))[0]}_pred.png"
                    )
                    visualize_result(img_path, prob, pred_class, true_class, output_path)
            
            results.append((img_path, prob, pred_class, true_class))
        
        # Calculate metrics if true labels are available
        true_labels = [true_class for _, _, _, true_class in results if true_class != "Unknown"]
        pred_labels = [pred_class for _, _, pred_class, true_class in results if true_class != "Unknown"]
        
        metrics = None
        if true_labels and len(true_labels) == len(pred_labels):
            metrics = calculate_metrics(true_labels, pred_labels)
        
        # Save results to CSV
        if batch_output_dir:
            import csv
            csv_path = os.path.join(batch_output_dir, "inference_results.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Include true label in the header if available
                if any(true_class != "Unknown" for _, _, _, true_class in results):
                    writer.writerow(['Image', 'Probability', 'Prediction', 'True Label', 'Correct'])
                    for img_path, prob, pred_class, true_class in results:
                        correct = "Yes" if pred_class == true_class else "No" if true_class != "Unknown" else "-"
                        writer.writerow([os.path.basename(img_path), f"{prob:.4f}", pred_class, 
                                        true_class if true_class != "Unknown" else "-", correct])
                else:
                    writer.writerow(['Image', 'Probability', 'Prediction'])
                    for img_path, prob, pred_class, _ in results:
                        writer.writerow([os.path.basename(img_path), f"{prob:.4f}", pred_class])
            
            # Also save metrics to a separate CSV if available
            if metrics:
                metrics_path = os.path.join(batch_output_dir, "performance_metrics.csv")
                with open(metrics_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Value'])
                    writer.writerow(['Accuracy', f"{metrics['accuracy']:.4f}"])
                    writer.writerow(['Precision', f"{metrics['precision']:.4f}"])
                    writer.writerow(['Recall', f"{metrics['recall']:.4f}"])
                    writer.writerow(['F1 Score', f"{metrics['f1']:.4f}"])
                    writer.writerow(['True Positives', metrics['true_positives']])
                    writer.writerow(['False Positives', metrics['false_positives']])
                    writer.writerow(['True Negatives', metrics['true_negatives']])
                    writer.writerow(['False Negatives', metrics['false_negatives']])
            
            print(f"Results saved to {csv_path}")
        
        # Print summary
        real_count = sum(1 for _, _, pred, _ in results if pred == "Real")
        fake_count = len(results) - real_count
        print(f"\nProcessed {len(results)} images from {input_folder_name}")
        if batch_output_dir:
            print(f"Results saved to {batch_output_dir}")
        print(f"Predicted Real: {real_count} ({real_count/len(results)*100:.1f}%)")
        print(f"Predicted Fake: {fake_count} ({fake_count/len(results)*100:.1f}%)")
        
        # Print metrics if available
        if metrics:
            print("\nPerformance Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Confusion Matrix:")
            print(f"                  | Predicted Real | Predicted Fake |")
            print(f"Actual Real      | {metrics['true_positives']:14d} | {metrics['false_negatives']:14d} |")
            print(f"Actual Fake      | {metrics['false_positives']:14d} | {metrics['true_negatives']:14d} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with GANFingerprint model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None, help='Directory to save output visualizations')
    parser.add_argument('--batch', action='store_true', help='Process a directory of images')
    parser.add_argument('--gradcam', action='store_true', help='Generate Grad-CAM visualizations')
    
    args = parser.parse_args()
    
    run_inference(args.checkpoint, args.input, args.output, args.batch, args.gradcam)