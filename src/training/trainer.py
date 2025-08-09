import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime


class Trainer:
    """Modern trainer for multimodal disaster response model"""
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.setup_scheduler()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create output directories
        self.output_dir = os.path.join(config['paths']['output_dir'], 
                                     datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config['training']['warmup_steps']
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs'] - self.config['training']['warmup_steps']
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config['training']['warmup_steps']]
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, input_ids, attention_mask)
            
            # Compute loss (multimodal + auxiliary losses)
            multimodal_loss = self.criterion(outputs['multimodal_logits'], labels)
            image_loss = self.criterion(outputs['image_logits'], labels)
            text_loss = self.criterion(outputs['text_logits'], labels)
            
            # Combined loss with weights
            loss = multimodal_loss + 0.3 * image_loss + 0.3 * text_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = outputs['predictions']
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, input_ids, attention_mask)
                
                # Compute loss
                loss = self.criterion(outputs['multimodal_logits'], labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = outputs['predictions']
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Store for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoints', 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation accuracy: {self.best_val_acc:.2f}%")
    
    def plot_metrics(self):
        """Plot training metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plots
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot([self.scheduler.get_last_lr()[0]] * len(self.train_losses))
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # Remove empty subplot
        ax4.remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, predictions, labels, epoch):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config['classes'],
                   yticklabels=self.config['classes'])
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'confusion_matrix_epoch_{epoch}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['training']['epochs']} epochs")
        print(f"Output directory: {self.output_dir}")
        
        # Initialize wandb if available
        try:
            wandb.init(
                project="disaster-response-multimodal",
                config=self.config,
                dir=self.output_dir
            )
            use_wandb = True
        except:
            print("Wandb not available, continuing without logging")
            use_wandb = False
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['training']['epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_predictions, val_labels = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Plot confusion matrix for best epochs
            if is_best:
                self.plot_confusion_matrix(val_predictions, val_labels, epoch)
                
                # Print classification report
                print("\nClassification Report:")
                print(classification_report(
                    val_labels, val_predictions,
                    target_names=self.config['classes'],
                    digits=4
                ))
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            # Plot metrics every 10 epochs
            if epoch % 10 == 0:
                self.plot_metrics()
        
        # Final plots and save
        self.plot_metrics()
        self.save_checkpoint(epoch, False)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        if use_wandb:
            wandb.finish()
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Results saved to: {self.output_dir}")
        
        return self.best_val_acc