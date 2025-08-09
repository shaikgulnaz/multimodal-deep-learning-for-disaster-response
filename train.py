#!/usr/bin/env python3
"""
Modern training script for multimodal disaster response model
"""

import os
import yaml
import torch
import argparse
from src.models.multimodal_model import MultimodalModel
from src.data.dataset import create_data_loaders
from src.training.trainer import Trainer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Disaster Response Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override data directory if provided
    if args.data_dir != 'data':
        config['paths']['data_dir'] = args.data_dir
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Configuration loaded from: {args.config}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Initializing model...")
    model = MultimodalModel(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Resume trainer state if checkpoint provided
    if args.resume:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_acc = checkpoint['best_val_acc']
        print(f"Resumed training from epoch {checkpoint['epoch']}")
    
    # Start training
    print("Starting training...")
    best_accuracy = trainer.train()
    
    print(f"Training completed! Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == '__main__':
    main()