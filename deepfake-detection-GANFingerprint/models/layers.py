"""
Custom layers and blocks for the GANFingerprint model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important regions in feature maps.
    This helps the model identify subtle GAN fingerprint artifacts.
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate attention map
        attn = self.conv(x)
        attn = self.sigmoid(attn)
        
        # Apply attention
        return x * attn

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight
        
    def forward(self, x):
        batch_size, C, height, width = x.shape
        
        # Reshape for matrix multiplication
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C'
        proj_key = self.key(x).view(batch_size, -1, height * width)  # B x C' x HW
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)  # B x HW x HW
        attention = F.softmax(energy, dim=-1)  # B x HW x HW
        
        # Apply attention
        proj_value = self.value(x).view(batch_size, -1, height * width)  # B x C x HW
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, height, width)  # B x C x H x W
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out

class FrequencyAwareness(nn.Module):
    """
    Module to enhance frequency domain awareness.
    GANs often leave artifacts in the frequency domain, so this module
    helps extract those signals.
    """
    def __init__(self, in_channels, out_channels):
        super(FrequencyAwareness, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Convert to frequency domain (approximation using DCT-like filters)
        # Note: Full DCT would be more computationally expensive
        # We use conv layers as a learnable frequency filter bank
        
        # First branch - spatial features
        spatial = self.relu(self.bn1(self.conv1(x)))
        spatial = self.bn2(self.conv2(spatial))
        
        # Second branch - high-pass filtering for frequency awareness
        avg_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        freq = x - avg_pool  # High-pass filtering
        freq = self.relu(self.bn1(self.conv1(freq)))
        freq = self.bn2(self.conv2(freq))
        
        # Combine branches
        return self.relu(spatial + freq)


class FingerprintBlock(nn.Module):
    """
    The core block for GANFingerprint detection.
    Combines frequency awareness with spatial attention.
    """
    def __init__(self, in_channels, out_channels):
        super(FingerprintBlock, self).__init__()
        self.freq_aware = FrequencyAwareness(in_channels, out_channels)
        self.spatial_attn = SpatialAttention(out_channels)
        self.dropout = nn.Dropout2d(p=0.1)  # Feature dropout for regularization
    
    def forward(self, x):
        x = self.freq_aware(x)
        x = self.spatial_attn(x)
        x = self.dropout(x)
        return x
    
# Frequency domain analysis
class DCTLayer(nn.Module):
    """Perform Discrete Cosine Transform approximation using FFT"""
    
    def __init__(self):
        super(DCTLayer, self).__init__()
    
    def forward(self, x):
        # Use FFT to approximate DCT
        x_fft = torch.fft.rfft2(x)
        # Get magnitude spectrum (discard phase)
        x_magnitude = torch.abs(x_fft)
        # Apply log scaling to emphasize fine details
        x_magnitude = torch.log(x_magnitude + 1e-10)
        return x_magnitude


class FrequencyEncoder(nn.Module):
    """Encode frequency domain information using CNN layers"""
    
    def __init__(self, in_channels=3):
        super(FrequencyEncoder, self).__init__()
        self.dct = DCTLayer()
        
        # For RGB image, the FFT will have different dimensions
        # We process each channel separately and concatenate
        
        # CNN layers to process frequency information
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Output dimensions will be 256 channels
        self.out_channels = 256
    
    def forward(self, x):
        # Apply DCT to each channel and process separately
        batch_size, channels, height, width = x.shape
        
        # FFT produces complex output with specific shape
        # For real input of shape (B, C, H, W), rfft2 output is (B, C, H, W//2+1, 2)
        # where the last dimension represents real and imaginary parts
        
        # Apply DCT to each channel
        dct_features = []
        for c in range(channels):
            dct_c = self.dct(x[:, c:c+1, :, :])
            dct_features.append(dct_c)
            
        # Concatenate along channel dimension
        # Each dct_c has shape (B, 1, H, W//2+1)
        x_dct = torch.cat(dct_features, dim=1)
        
        # Process through CNN
        x_freq = self.conv1(x_dct)
        x_freq = self.conv2(x_freq)
        x_freq = self.conv3(x_freq)
        
        return x_freq

