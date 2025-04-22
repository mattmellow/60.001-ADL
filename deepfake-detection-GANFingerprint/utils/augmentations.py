import random
import numpy as np
import torch
from io import BytesIO
from PIL import Image, ImageFilter

class JPEGCompression:
    """
    Simulate JPEG compression artifacts, common in real-world images.
    Deepfakes often lack these compression patterns.
    """
    def __init__(self, quality_range=(60, 95)):
        self.quality_range = quality_range
        
    def __call__(self, img):
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class AddNoiseGaussian:
    """
    Add Gaussian noise to simulate camera sensor noise.
    GAN-generated images often have different noise distributions.
    """
    def __init__(self, mean=0., std_range=(0.01, 0.05)):
        self.mean = mean
        self.std_range = std_range
        
    def __call__(self, img):
        std = random.uniform(self.std_range[0], self.std_range[1])
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1)  # Convert to CxHxW
        
        # Add noise
        noise = torch.randn_like(img_tensor) * std + self.mean
        noisy_tensor = torch.clamp(img_tensor + noise, 0, 1)
        
        # Convert back to PIL
        noisy_tensor = noisy_tensor.permute(1, 2, 0)  # Convert to HxWxC
        noisy_img = Image.fromarray((noisy_tensor.numpy() * 255).astype(np.uint8))
        return noisy_img

class VariableBlur:
    """
    Apply varying levels of blur to simulate camera focus issues.
    Helps model learn features robust to blur variations.
    """
    def __init__(self, radius_range=(0.1, 2.0)):
        self.radius_range = radius_range
        
    def __call__(self, img):
        radius = random.uniform(self.radius_range[0], self.radius_range[1])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
        
class ColorQuantization:
    """
    Simulate color quantization artifacts from image compression.
    """
    def __init__(self, bits_range=(5, 8)):
        self.bits_range = bits_range
        
    def __call__(self, img):
        bits = random.randint(self.bits_range[0], self.bits_range[1])
        img_np = np.array(img)
        
        # Quantize
        img_np = img_np >> (8 - bits)
        img_np = img_np << (8 - bits)
        
        return Image.fromarray(img_np.astype(np.uint8))