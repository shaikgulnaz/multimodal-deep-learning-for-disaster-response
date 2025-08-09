#!/usr/bin/env python3
"""
Prediction script for multimodal disaster response model
"""

import os
import yaml
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from src.models.multimodal_model import MultimodalModel


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_image(image_path, config):
    """Preprocess image for inference"""
    image_size = config['data']['image_size']
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def preprocess_text(text, config, tokenizer):
    """Preprocess text for inference"""
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=config['data']['max_text_length'],
        return_tensors='pt'
    )
    return inputs['input_ids'], inputs['attention_mask']


def predict_single(model, image_path, text, config, device):
    """Make prediction on a single image-text pair"""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model'])
    
    # Preprocess inputs
    image = preprocess_image(image_path, config).to(device)
    input_ids, attention_mask = preprocess_text(text, config, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image, input_ids, attention_mask)
        
        # Get probabilities
        probs = torch.softmax(outputs['multimodal_logits'], dim=1)
        image_probs = torch.softmax(outputs['image_logits'], dim=1)
        text_probs = torch.softmax(outputs['text_logits'], dim=1)
        
        # Get predictions
        multimodal_pred = outputs['predictions'].item()
        image_pred = torch.argmax(outputs['image_logits'], dim=1).item()
        text_pred = torch.argmax(outputs['text_logits'], dim=1).item()
    
    return {
        'multimodal': {
            'prediction': multimodal_pred,
            'class': config['classes'][multimodal_pred],
            'confidence': probs[0][multimodal_pred].item(),
            'probabilities': probs[0].cpu().numpy()
        },
        'image_only': {
            'prediction': image_pred,
            'class': config['classes'][image_pred],
            'confidence': image_probs[0][image_pred].item(),
            'probabilities': image_probs[0].cpu().numpy()
        },
        'text_only': {
            'prediction': text_pred,
            'class': config['classes'][text_pred],
            'confidence': text_probs[0][text_pred].item(),
            'probabilities': text_probs[0].cpu().numpy()
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Predict with Multimodal Disaster Response Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--text', type=str, required=True,
                       help='Input text description')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Create model
    model = MultimodalModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded successfully")
    print(f"Image: {args.image}")
    print(f"Text: {args.text}")
    print("-" * 50)
    
    # Make prediction
    results = predict_single(model, args.image, args.text, config, device)
    
    # Print results
    print("PREDICTION RESULTS")
    print("=" * 50)
    
    print(f"\nMultimodal Prediction:")
    print(f"  Class: {results['multimodal']['class']}")
    print(f"  Confidence: {results['multimodal']['confidence']:.4f}")
    
    print(f"\nImage-only Prediction:")
    print(f"  Class: {results['image_only']['class']}")
    print(f"  Confidence: {results['image_only']['confidence']:.4f}")
    
    print(f"\nText-only Prediction:")
    print(f"  Class: {results['text_only']['class']}")
    print(f"  Confidence: {results['text_only']['confidence']:.4f}")
    
    print(f"\nClass Probabilities (Multimodal):")
    for i, (class_name, prob) in enumerate(zip(config['classes'], results['multimodal']['probabilities'])):
        print(f"  {class_name}: {prob:.4f}")


if __name__ == '__main__':
    main()