#!/usr/bin/env python3
"""
Evaluation script for multimodal disaster response model
"""

import os
import yaml
import torch
import argparse
from src.models.multimodal_model import MultimodalModel
from src.data.dataset import create_data_loaders
from src.evaluation.evaluator import ModelEvaluator


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multimodal Disaster Response Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override paths if provided
    if args.data_dir != 'data':
        config['paths']['data_dir'] = args.data_dir
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Initializing model...")
    model = MultimodalModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown'):.2f}%")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader, config, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluator.evaluate(save_dir=args.output_dir)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'evaluation_report.md')
    report = evaluator.generate_report(results, report_path)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Multimodal Accuracy: {results['multimodal_accuracy']:.2f}%")
    print(f"Image-only Accuracy: {results['image_accuracy']:.2f}%")
    print(f"Text-only Accuracy: {results['text_accuracy']:.2f}%")
    print(f"Macro F1-Score: {results['macro_avg']['f1_score']:.4f}")
    
    if results['multimodal_auc']:
        print(f"Multimodal AUC: {results['multimodal_auc']:.4f}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")
    print(f"Evaluation report: {report_path}")


if __name__ == '__main__':
    main()