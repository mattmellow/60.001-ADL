"""
Grad-CAM implementation for the GANFingerprint model
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention in the GANFingerprint model.
    """
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM with model and target layer
        
        Args:
            model: The trained model
            target_layer: The target layer for visualization (typically last convolutional layer)
        """
        self.model = model
        self.model.eval()
        
        # Get the target layer
        self.target_layer = target_layer
        
        # Register hooks
        self.gradients = None
        self.activations = None
        
        # Register forward hook
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        
        # Register backward hook
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save the activations of the target layer"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save the gradients of the target layer"""
        self.gradients = grad_output[0].detach()
    
    def remove_hooks(self):
        """Remove the hooks to free memory"""
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            x: Input tensor (must be a single image, not a batch)
            class_idx: Index of the class to generate CAM for (None for binary classification)
            
        Returns:
            Tuple of (raw logits, normalized heatmap)
        """
        # Forward pass
        x = x.unsqueeze(0)  # Add batch dimension if not present
        output = self.model(x)
        
        # For binary classification model, we interpret the output as logits
        # The gradient needs to be calculated with respect to the output
        if class_idx is None:
            # For binary classification, we'll get gradients w.r.t. the positive class (real)
            self.model.zero_grad()
            output.backward(retain_graph=True)
        else:
            # For multi-class models (unlikely in your case)
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1
            output.backward(gradient=one_hot, retain_graph=True)
        
        # Get the gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Take the average of gradients along each channel
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weight the channels by corresponding gradients
        cam = torch.sum(weights * activations, dim=1).squeeze(0)
        
        # Apply ReLU to the heatmap
        cam = F.relu(cam)
        
        # Normalize the heatmap
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Resize to input size and convert to numpy
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                           size=x.shape[2:],
                           mode='bilinear', 
                           align_corners=False)
        
        # Convert to numpy array and normalize
        cam = cam.squeeze().cpu().detach().numpy()
        
        return output.squeeze().item(), cam


def get_gradcam_layer(model):
    """
    Helper function to get the last convolutional layer of the model
    
    Args:
        model: The FingerprintNet model
        
    Returns:
        The last convolutional layer (target layer for Grad-CAM)
    """
    # First, print model structure to debug
    # print("Model structure:")
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         print(f"Found Conv2d layer: {name}")
    
    # Try to find the last convolutional layer
    target_layer = None
    last_conv_name = None
    
    # Approach 1: Check if model has backbone attribute
    if hasattr(model, 'backbone'):
        # For ResNet-based models
        if hasattr(model.backbone, 'layer4'):
            target_layer = model.backbone.layer4[-1]
            # print("Found target layer: model.backbone.layer4[-1]")
        # For EfficientNet
        elif hasattr(model.backbone, 'features'):
            # Get the last Conv2d layer from features
            for layer in reversed(model.backbone.features):
                if isinstance(layer, torch.nn.Conv2d):
                    target_layer = layer
                    # print("Found target layer in model.backbone.features")
                    break
    
    # Approach 2: If no backbone, directly search for the last Conv2d in the model
    if target_layer is None:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                last_conv_name = name
        
        # if target_layer is not None:
        #     print(f"Using last Conv2d layer found: {last_conv_name}")
    
    # Fallback approach: If still no layer found, try specific common structures
    if target_layer is None:
        # Check for base_model attribute (common in transfer learning)
        if hasattr(model, 'base_model'):
            base_model = model.base_model
            for name, module in base_model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    last_conv_name = f"base_model.{name}"
            # if target_layer is not None:
            #     print(f"Using Conv2d layer from base_model: {last_conv_name}")
    
    # Final fallback - try to access specific paths that might exist
    if target_layer is None:
        try:
            if hasattr(model, 'features'):
                # Common in VGG, AlexNet, etc.
                for i in reversed(range(len(model.features))):
                    if isinstance(model.features[i], torch.nn.Conv2d):
                        target_layer = model.features[i]
                        # print(f"Found target layer: model.features[{i}]")
                        break
        except:
            pass
    
    if target_layer is None:
        raise ValueError("Could not find a suitable target layer for Grad-CAM")
    
    return target_layer


def generate_gradcam(model, image_tensor, orig_image):
    """
    Generate and visualize Grad-CAM for a given image
    
    Args:
        model: The FingerprintNet model
        image_tensor: Preprocessed image tensor
        orig_image: Original PIL image for visualization
        
    Returns:
        Tuple of (raw_logit, normalized heatmap as numpy array, visualization image)
    """
    # Get the target layer for Grad-CAM
    target_layer = get_gradcam_layer(model)
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate Grad-CAM
    raw_logit, heatmap = grad_cam(image_tensor)
    
    # Clean up hooks
    grad_cam.remove_hooks()
    
    # Create visualization
    # Convert PIL Image to numpy array
    img_np = np.array(orig_image)
    
    # Create color map
    heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Overlay heatmap on original image
    superimposed = (0.7 * heatmap_colored + 0.3 * img_np / 255.0)
    
    # Ensure values are in valid range
    superimposed = np.clip(superimposed, 0, 1)
    
    return raw_logit, heatmap, superimposed


def visualize_gradcam(image_path, model, transform, output_path=None):
    """
    Visualize Grad-CAM for a given image with improved layout
    
    Args:
        image_path: Path to input image
        model: The FingerprintNet model
        transform: Image transformation pipeline
        output_path: Path to save visualization (optional)
        
    Returns:
        Tuple of (raw_logit, prediction class, calibrated probability)
    """
    # Load and preprocess the image
    orig_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(orig_image).to(next(model.parameters()).device)
    
    # Generate Grad-CAM
    raw_logit, heatmap, superimposed = generate_gradcam(model, image_tensor, orig_image)
    
    # Calculate the prediction
    orig_prob = torch.sigmoid(torch.tensor(raw_logit)).item()
    
    # Apply calibration for fake images (same as in predict_image_calibrated)
    if orig_prob < 0.5:  # Predicted as fake
        calibrated_prob = 1.0 - (2.0 * orig_prob)
    else:  # Predicted as real
        calibrated_prob = orig_prob
    
    pred_class = "Real" if orig_prob >= 0.5 else "Fake"
    
    # Extract true label if available
    true_class = None
    if image_path:
        import os
        basename = os.path.basename(image_path).lower()
        if 'real' in basename:
            true_class = "Real"
        elif 'fake' in basename:
            true_class = "Fake"
    
    # Create visualization
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
    
    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    return raw_logit, pred_class, calibrated_prob