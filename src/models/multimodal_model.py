import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_model import ImageEncoder
from .text_model import TextEncoder


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism"""
    
    def __init__(self, image_dim, text_dim, hidden_dim=512):
        super(AttentionFusion, self).__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, image_features, text_features):
        # Project to same dimension
        img_proj = self.image_proj(image_features).unsqueeze(1)  # [B, 1, H]
        txt_proj = self.text_proj(text_features).unsqueeze(1)    # [B, 1, H]
        
        # Concatenate for attention
        combined = torch.cat([img_proj, txt_proj], dim=1)  # [B, 2, H]
        
        # Self-attention
        attended, _ = self.attention(combined, combined, combined)
        attended = self.norm(attended)
        
        # Global average pooling
        fused = attended.mean(dim=1)  # [B, H]
        return fused


class BilinearFusion(nn.Module):
    """Bilinear fusion mechanism"""
    
    def __init__(self, image_dim, text_dim, hidden_dim=512):
        super(BilinearFusion, self).__init__()
        self.bilinear = nn.Bilinear(image_dim, text_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, image_features, text_features):
        fused = self.bilinear(image_features, text_features)
        return self.norm(fused)


class MultimodalModel(nn.Module):
    """Multimodal model for disaster response classification"""
    
    def __init__(self, config):
        super(MultimodalModel, self).__init__()
        self.config = config
        
        # Initialize encoders
        self.image_encoder = ImageEncoder(
            model_name=config['model']['image_model'],
            num_classes=config['model']['num_classes']
        )
        
        self.text_encoder = TextEncoder(
            model_name=config['model']['text_model'],
            num_classes=config['model']['num_classes'],
            max_length=config['data']['max_text_length']
        )
        
        # Fusion mechanism
        fusion_method = config['model']['fusion_method']
        image_dim = self.image_encoder.feature_dim
        text_dim = self.text_encoder.feature_dim
        
        if fusion_method == "concatenation":
            self.fusion = nn.Identity()
            fusion_dim = image_dim + text_dim
        elif fusion_method == "attention":
            self.fusion = AttentionFusion(image_dim, text_dim)
            fusion_dim = 512
        elif fusion_method == "bilinear":
            self.fusion = BilinearFusion(image_dim, text_dim)
            fusion_dim = 512
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(256, config['model']['num_classes'])
        )
        
    def forward(self, images, input_ids, attention_mask):
        # Extract features
        image_features, image_logits = self.image_encoder(images)
        text_features, text_logits = self.text_encoder(input_ids, attention_mask)
        
        # Fusion
        if self.config['model']['fusion_method'] == "concatenation":
            fused_features = torch.cat([image_features, text_features], dim=1)
        else:
            fused_features = self.fusion(image_features, text_features)
        
        # Final classification
        multimodal_logits = self.classifier(fused_features)
        
        return {
            'image_features': image_features,
            'text_features': text_features,
            'fused_features': fused_features,
            'image_logits': image_logits,
            'text_logits': text_logits,
            'multimodal_logits': multimodal_logits,
            'predictions': torch.argmax(multimodal_logits, dim=1)
        }