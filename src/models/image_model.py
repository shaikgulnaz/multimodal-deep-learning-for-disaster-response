import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel, ViTConfig


class ImageEncoder(nn.Module):
    """Modern image encoder supporting multiple architectures"""
    
    def __init__(self, model_name="resnet50", num_classes=6, pretrained=True):
        super(ImageEncoder, self).__init__()
        self.model_name = model_name
        
        if model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == "efficientnet-b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == "vit-base-patch16-224":
            config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
            self.feature_dim = config.hidden_size
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        if self.model_name == "vit-base-patch16-224":
            outputs = self.backbone(pixel_values=x)
            features = outputs.last_hidden_state[:, 0]  # CLS token
        else:
            features = self.backbone(x)
            
        features = self.dropout(features)
        logits = self.classifier(features)
        return features, logits


class ImageModel(nn.Module):
    """Standalone image classification model"""
    
    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.encoder = ImageEncoder(
            model_name=config['model']['image_model'],
            num_classes=config['model']['num_classes'],
            pretrained=True
        )
        
    def forward(self, images):
        features, logits = self.encoder(images)
        return {
            'features': features,
            'logits': logits,
            'predictions': torch.argmax(logits, dim=1)
        }