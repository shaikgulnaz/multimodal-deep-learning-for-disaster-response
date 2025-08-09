import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class TextEncoder(nn.Module):
    """Modern text encoder using transformer models"""
    
    def __init__(self, model_name="bert-base-uncased", num_classes=6, max_length=128):
        super(TextEncoder, self).__init__()
        self.model_name = model_name
        self.max_length = max_length
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.feature_dim = self.config.hidden_size
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token representation
        features = outputs.last_hidden_state[:, 0]
        features = self.dropout(features)
        logits = self.classifier(features)
        
        return features, logits
    
    def tokenize(self, texts):
        """Tokenize text inputs"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )


class TextModel(nn.Module):
    """Standalone text classification model"""
    
    def __init__(self, config):
        super(TextModel, self).__init__()
        self.encoder = TextEncoder(
            model_name=config['model']['text_model'],
            num_classes=config['model']['num_classes'],
            max_length=config['data']['max_text_length']
        )
        
    def forward(self, input_ids, attention_mask):
        features, logits = self.encoder(input_ids, attention_mask)
        return {
            'features': features,
            'logits': logits,
            'predictions': torch.argmax(logits, dim=1)
        }