import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
import timm
from .model_embedder import HybridEmbed
from torchvision.models import resnet18


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)

        # Remove the final layers (avgpool + fc)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Outputs feature map (e.g., 512x7x7)

    def forward(self, x):
        return self.features(x)

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)

class GenConViTED(nn.Module):
    def __init__(self, config, pretrained=True):
        super(GenConViTED, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.backbone = timm.create_model(config['model']['backbone'], pretrained=pretrained)
        self.embedder = timm.create_model(config['model']['embedder'], pretrained=pretrained)
        self.backbone.patch_embed = HybridEmbed(self.embedder, img_size=config['img_size'], embed_dim=768)

        self.num_features = self.backbone.fc.out_features * 2
        self.fc = nn.Linear(self.num_features, self.num_features//4)
        self.fc2 = nn.Linear(self.num_features // 4, 2)
        self.relu = nn.GELU()

    def forward(self, images):

        encimg = self.encoder(images)
        decimg = self.decoder(encimg)

        x1 = self.backbone(decimg)
        x2 = self.backbone(images)

        x = torch.cat((x1,x2), dim=1)

        x = self.fc2(self.relu(self.fc(self.relu(x))))


        return x.squeeze()