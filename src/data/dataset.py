import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import json
import pandas as pd
from sklearn.model_selection import train_test_split


class DisasterDataset(Dataset):
    """Dataset for multimodal disaster response classification"""
    
    def __init__(self, data_dir, split='train', config=None, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.config = config
        self.transform = transform
        
        # Load class mapping
        self.classes = config['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model'])
        
        # Load data
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load samples from directory structure"""
        samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            images_dir = os.path.join(class_dir, 'images')
            text_dir = os.path.join(class_dir, 'text')
            
            if not (os.path.exists(images_dir) and os.path.exists(text_dir)):
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                # Find corresponding text file
                base_name = os.path.splitext(img_file)[0]
                text_file = os.path.join(text_dir, f"{base_name}.txt")
                
                if os.path.exists(text_file):
                    samples.append({
                        'image_path': os.path.join(images_dir, img_file),
                        'text_path': text_file,
                        'label': self.class_to_idx[class_name],
                        'class_name': class_name
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load text
        with open(sample['text_path'], 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Tokenize text
        text_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config['data']['max_text_length'],
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'class_name': sample['class_name']
        }


def get_transforms(config, split='train'):
    """Get image transforms for different splits"""
    image_size = config['data']['image_size']
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(config):
    """Create data loaders for train, validation, and test sets"""
    data_dir = config['paths']['data_dir']
    
    # Create transforms
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')
    
    # Create datasets
    train_dataset = DisasterDataset(
        data_dir=os.path.join(data_dir, 'train'),
        split='train',
        config=config,
        transform=train_transform
    )
    
    val_dataset = DisasterDataset(
        data_dir=os.path.join(data_dir, 'val'),
        split='val',
        config=config,
        transform=val_transform
    )
    
    test_dataset = DisasterDataset(
        data_dir=os.path.join(data_dir, 'test'),
        split='test',
        config=config,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def split_dataset(data_dir, output_dir, config):
    """Split dataset into train/val/test sets"""
    import shutil
    from collections import defaultdict
    
    # Collect all samples by class
    class_samples = defaultdict(list)
    
    for class_name in config['classes']:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        images_dir = os.path.join(class_dir, 'images')
        text_dir = os.path.join(class_dir, 'text')
        
        if not (os.path.exists(images_dir) and os.path.exists(text_dir)):
            continue
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            text_file = os.path.join(text_dir, f"{base_name}.txt")
            
            if os.path.exists(text_file):
                class_samples[class_name].append({
                    'image_file': img_file,
                    'text_file': f"{base_name}.txt",
                    'image_path': os.path.join(images_dir, img_file),
                    'text_path': text_file
                })
    
    # Split each class
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    
    for split in ['train', 'val', 'test']:
        for class_name in config['classes']:
            os.makedirs(os.path.join(output_dir, split, class_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, class_name, 'text'), exist_ok=True)
    
    for class_name, samples in class_samples.items():
        # Split samples
        train_samples, temp_samples = train_test_split(
            samples, train_size=train_split, random_state=42
        )
        val_samples, test_samples = train_test_split(
            temp_samples, train_size=val_split/(1-train_split), random_state=42
        )
        
        # Copy files to respective directories
        for split, split_samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            for sample in split_samples:
                # Copy image
                dst_img = os.path.join(output_dir, split, class_name, 'images', sample['image_file'])
                shutil.copy2(sample['image_path'], dst_img)
                
                # Copy text
                dst_txt = os.path.join(output_dir, split, class_name, 'text', sample['text_file'])
                shutil.copy2(sample['text_path'], dst_txt)
    
    print(f"Dataset split completed. Files saved to {output_dir}")