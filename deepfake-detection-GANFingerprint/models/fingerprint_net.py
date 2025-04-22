"""
GANFingerprint model architecture for deepfake detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .layers import FingerprintBlock, FrequencyEncoder, SelfAttention
import config


class FingerprintNet(nn.Module):
    """GANFingerprint network with multi-layer feature fusion."""
    
    def __init__(self, backbone=config.BACKBONE, pretrained=True):
        super(FingerprintNet, self).__init__()
        
        # Load pre-trained backbone
        if backbone == 'resnet34':
            if pretrained:
                base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                base_model = models.resnet34(weights=None)
            
            # Extract different parts of the ResNet backbone
            self.backbone_low = nn.Sequential(*list(base_model.children())[:5])  # Until layer1
            self.backbone_mid = nn.Sequential(*list(base_model.children())[5:6])  # layer2
            self.backbone_high = nn.Sequential(*list(base_model.children())[6:8])  # layer3 & layer4
            
            # Feature dimensions from each part
            self.low_dims = 64  # After conv1 and maxpool
            self.mid_dims = 128  # After layer2
            self.high_dims = 512  # After layer4
        else:
            raise ValueError(f"Backbone {backbone} not supported for feature fusion")
        
        # Adaptation layers to make feature dimensions compatible
        self.low_adapter = nn.Sequential(
            nn.Conv2d(self.low_dims, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.mid_adapter = nn.Sequential(
            nn.Conv2d(self.mid_dims, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.high_adapter = nn.Sequential(
            nn.Conv2d(self.high_dims, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(128 * 3, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Add frequency domain encoder
        # self.freq_encoder = FrequencyEncoder(in_channels=3)
        self.freq_encoder = FrequencyEncoder(in_channels=3)
        
        # Modify fusion layer to include frequency features
        # If your current fusion combines 3 streams (low, mid, high) with 128 channels each
        self.fusion_layer = nn.Sequential(
            # Now include frequency features (256 channels) along with spatial features (3*128 channels)
            nn.Conv2d(128*3 + self.freq_encoder.out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Add fingerprint blocks after fusion
        self.fingerprint_block = FingerprintBlock(256, 256)

        self.attention = SelfAttention(256)  
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(256, config.EMBEDDING_DIM),
            nn.BatchNorm1d(config.EMBEDDING_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE)
        )
        
        class EnhancedClassifier(nn.Module):
            def __init__(self, embedding_dim, dropout_rate):
                super(EnhancedClassifier, self).__init__()
                
                self.layer1 = nn.Sequential(
                    nn.Linear(embedding_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.Dropout(dropout_rate)
                )
                
                self.layer2 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.Dropout(dropout_rate)
                )
                
                # Residual connection adapter
                self.residual_adapter = nn.Linear(embedding_dim, 256)
                
                self.layer3 = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.Dropout(dropout_rate * 0.8)
                )
                
                self.classifier = nn.Linear(128, 1)
            
            def forward(self, x):
                # First layer
                out1 = self.layer1(x)
                
                # Second layer with residual connection
                out2 = self.layer2(out1)
                res = self.residual_adapter(x)
                out2 = out2 + res  # Residual connection
                
                # Final layers
                out3 = self.layer3(out2)
                return self.classifier(out3)
            
        self.classifier = EnhancedClassifier(config.EMBEDDING_DIM, config.DROPOUT_RATE)
        
        # Initialize weights for new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights deterministically"""
        # Use a fixed seed for weight initialization
        torch.manual_seed(42)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        # Reset seed after initialization
        torch.manual_seed(torch.initial_seed())
        
    def forward(self, x):
        # Extract multi-level spatial features
        low_features = self.backbone_low(x)
        mid_features = self.backbone_mid(low_features)
        high_features = self.backbone_high(mid_features)
        
        # Adapt features to common dimensions
        low_adapted = self.low_adapter(F.adaptive_avg_pool2d(low_features, high_features.shape[2:]))
        mid_adapted = self.mid_adapter(F.adaptive_avg_pool2d(mid_features, high_features.shape[2:]))
        high_adapted = self.high_adapter(high_features)
        
        # Extract frequency domain features
        freq_features = self.freq_encoder(x)
        
        # Resize frequency features to match spatial features if needed
        if freq_features.shape[2:] != high_features.shape[2:]:
            freq_features = F.interpolate(
                freq_features, 
                size=high_features.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # Fuse all features: spatial (low, mid, high) and frequency
        fused = torch.cat([low_adapted, mid_adapted, high_adapted, freq_features], dim=1)
        fused = self.fusion_layer(fused)
        # Attention 
        fused = self.attention(fused)
        
        # Continue with the rest of your forward pass
        features = self.fingerprint_block(fused)
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        
        return logits.squeeze()


    def fingerprint_distance(self, x1, x2):
        """
        Calculate fingerprint distance between two batches of images.
        Useful for analyzing similarities between real and fake images.
        """
        # Get embeddings
        emb1 = self.extract_features(x1)
        emb2 = self.extract_features(x2)
        
        # Normalize embeddings
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        
        # Calculate cosine similarity (or distance)
        cos_sim = torch.mm(emb1, emb2.transpose(0, 1))
        
        return cos_sim