import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, test_loader, config, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.classes = config['classes']
        
    def evaluate(self, save_dir=None):
        """Comprehensive evaluation of the model"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        # Individual modality predictions
        image_predictions = []
        text_predictions = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating')
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, input_ids, attention_mask)
                
                # Get predictions and probabilities
                multimodal_probs = torch.softmax(outputs['multimodal_logits'], dim=1)
                image_preds = torch.argmax(outputs['image_logits'], dim=1)
                text_preds = torch.argmax(outputs['text_logits'], dim=1)
                
                # Store results
                all_predictions.extend(outputs['predictions'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(multimodal_probs.cpu().numpy())
                image_predictions.extend(image_preds.cpu().numpy())
                text_predictions.extend(text_preds.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        image_predictions = np.array(image_predictions)
        text_predictions = np.array(text_predictions)
        
        # Calculate metrics
        results = self._calculate_metrics(
            all_labels, all_predictions, all_probabilities,
            image_predictions, text_predictions
        )
        
        # Generate visualizations
        if save_dir:
            self._generate_visualizations(
                all_labels, all_predictions, all_probabilities,
                image_predictions, text_predictions, save_dir
            )
            
            # Save results
            with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _calculate_metrics(self, labels, predictions, probabilities, 
                          image_preds, text_preds):
        """Calculate comprehensive metrics"""
        results = {}
        
        # Overall accuracy
        results['multimodal_accuracy'] = np.mean(predictions == labels) * 100
        results['image_accuracy'] = np.mean(image_preds == labels) * 100
        results['text_accuracy'] = np.mean(text_preds == labels) * 100
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, labels=range(len(self.classes))
        )
        
        results['per_class_metrics'] = {}
        for i, class_name in enumerate(self.classes):
            results['per_class_metrics'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        results['macro_avg'] = {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1_score': float(macro_f1)
        }
        
        results['weighted_avg'] = {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1_score': float(weighted_f1)
        }
        
        # AUC scores (if applicable)
        try:
            results['multimodal_auc'] = float(roc_auc_score(
                labels, probabilities, multi_class='ovr', average='macro'
            ))
        except:
            results['multimodal_auc'] = None
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(labels, predictions).tolist()
        
        return results
    
    def _generate_visualizations(self, labels, predictions, probabilities,
                               image_preds, text_preds, save_dir):
        """Generate evaluation visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(labels, predictions, save_dir)
        
        # 2. Per-class performance comparison
        self._plot_modality_comparison(labels, predictions, image_preds, text_preds, save_dir)
        
        # 3. Class distribution
        self._plot_class_distribution(labels, save_dir)
        
        # 4. Prediction confidence distribution
        self._plot_confidence_distribution(probabilities, predictions, labels, save_dir)
    
    def _plot_confusion_matrix(self, labels, predictions, save_dir):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Multimodal Model', fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   cbar_kws={'label': 'Proportion'})
        plt.title('Normalized Confusion Matrix - Multimodal Model', fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_modality_comparison(self, labels, multimodal_preds, image_preds, text_preds, save_dir):
        """Compare performance across modalities"""
        modalities = ['Image Only', 'Text Only', 'Multimodal']
        predictions_list = [image_preds, text_preds, multimodal_preds]
        
        # Calculate per-class F1 scores
        f1_scores = {}
        for i, class_name in enumerate(self.classes):
            f1_scores[class_name] = []
            class_mask = labels == i
            
            for preds in predictions_list:
                if np.sum(class_mask) > 0:
                    class_labels = labels[class_mask]
                    class_preds = preds[class_mask]
                    _, _, f1, _ = precision_recall_fscore_support(
                        class_labels, class_preds, average='binary', pos_label=i, zero_division=0
                    )
                    f1_scores[class_name].append(f1)
                else:
                    f1_scores[class_name].append(0)
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(15, 8))
        
        x = np.arange(len(self.classes))
        width = 0.25
        
        for i, modality in enumerate(modalities):
            scores = [f1_scores[class_name][i] for class_name in self.classes]
            ax.bar(x + i * width, scores, width, label=modality, alpha=0.8)
        
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Per-Class F1 Score Comparison Across Modalities', fontsize=14, pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'modality_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_distribution(self, labels, save_dir):
        """Plot class distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar([self.classes[i] for i in unique], counts, alpha=0.8, color='skyblue')
        plt.title('Test Set Class Distribution', fontsize=14, pad=20)
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, probabilities, predictions, labels, save_dir):
        """Plot prediction confidence distribution"""
        # Get max probabilities (confidence scores)
        confidences = np.max(probabilities, axis=1)
        correct_mask = predictions == labels
        
        plt.figure(figsize=(12, 6))
        
        # Plot histograms for correct and incorrect predictions
        plt.hist(confidences[correct_mask], bins=30, alpha=0.7, label='Correct Predictions', 
                color='green', density=True)
        plt.hist(confidences[~correct_mask], bins=30, alpha=0.7, label='Incorrect Predictions', 
                color='red', density=True)
        
        plt.xlabel('Prediction Confidence', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Prediction Confidence Distribution', fontsize=14, pad=20)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        correct_mean = np.mean(confidences[correct_mask])
        incorrect_mean = np.mean(confidences[~correct_mask])
        plt.axvline(correct_mean, color='green', linestyle='--', alpha=0.8, 
                   label=f'Correct Mean: {correct_mean:.3f}')
        plt.axvline(incorrect_mean, color='red', linestyle='--', alpha=0.8,
                   label=f'Incorrect Mean: {incorrect_mean:.3f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results, save_path):
        """Generate a comprehensive evaluation report"""
        report = []
        report.append("# Multimodal Disaster Response Model - Evaluation Report\n")
        report.append(f"Generated on: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall Performance
        report.append("## Overall Performance\n")
        report.append(f"- **Multimodal Accuracy**: {results['multimodal_accuracy']:.2f}%")
        report.append(f"- **Image-only Accuracy**: {results['image_accuracy']:.2f}%")
        report.append(f"- **Text-only Accuracy**: {results['text_accuracy']:.2f}%")
        if results['multimodal_auc']:
            report.append(f"- **Multimodal AUC**: {results['multimodal_auc']:.4f}")
        report.append("")
        
        # Macro Averages
        report.append("## Macro Averages\n")
        report.append(f"- **Precision**: {results['macro_avg']['precision']:.4f}")
        report.append(f"- **Recall**: {results['macro_avg']['recall']:.4f}")
        report.append(f"- **F1-Score**: {results['macro_avg']['f1_score']:.4f}")
        report.append("")
        
        # Per-class Performance
        report.append("## Per-Class Performance\n")
        report.append("| Class | Precision | Recall | F1-Score | Support |")
        report.append("|-------|-----------|--------|----------|---------|")
        
        for class_name, metrics in results['per_class_metrics'].items():
            report.append(f"| {class_name} | {metrics['precision']:.4f} | "
                         f"{metrics['recall']:.4f} | {metrics['f1_score']:.4f} | "
                         f"{metrics['support']} |")
        
        report.append("")
        
        # Model Configuration
        report.append("## Model Configuration\n")
        report.append(f"- **Image Model**: {self.config['model']['image_model']}")
        report.append(f"- **Text Model**: {self.config['model']['text_model']}")
        report.append(f"- **Fusion Method**: {self.config['model']['fusion_method']}")
        report.append(f"- **Number of Classes**: {self.config['model']['num_classes']}")
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)