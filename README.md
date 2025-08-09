# Multimodal Deep Learning for Disaster Response (Modernized)

A modern implementation of multimodal deep learning for disaster response classification using PyTorch and Transformers. This project combines image and text data to classify disaster-related social media posts into different damage categories.

## 🚀 Features

- **Modern Architecture**: Built with PyTorch and Hugging Face Transformers
- **Multiple Fusion Methods**: Concatenation, attention-based, and bilinear fusion
- **Flexible Backbones**: Support for ResNet, EfficientNet, Vision Transformer for images; BERT, RoBERTa, DistilBERT for text
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and model comparison
- **Easy to Use**: Simple command-line interface for training, evaluation, and prediction
- **Production Ready**: Proper logging, checkpointing, and model deployment utilities

## 📊 Dataset

The model works with the multimodal dataset from the original paper:
- **Classes**: 6 disaster categories (damaged_infrastructure, damaged_nature, fires, flood, human_damage, non_damage)
- **Modalities**: Images and corresponding text descriptions
- **Source**: Social media posts (Twitter, Instagram)

Dataset available at: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Multimodal+Damage+Identification+for+Humanitarian+Computing)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Install from source
```bash
git clone https://github.com/your-repo/multimodal-disaster-response.git
cd multimodal-disaster-response
pip install -e .
```

### Install dependencies only
```bash
pip install -r requirements.txt
```

## 📁 Data Preparation

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── damaged_infrastructure/
│   │   ├── images/
│   │   └── text/
│   ├── damaged_nature/
│   │   ├── images/
│   │   └── text/
│   └── ...
├── val/
└── test/
```

Each text file should have the same name as its corresponding image file (e.g., `image1.jpg` ↔ `image1.txt`).

## 🚀 Quick Start

### Training
```bash
# Train with default configuration
python train.py --config config/config.yaml --data-dir data/

# Train with custom settings
python train.py --config config/config.yaml --data-dir data/ --device cuda
```

### Evaluation
```bash
python evaluate.py --config config/config.yaml --checkpoint path/to/best_model.pth --data-dir data/
```

### Single Prediction
```bash
python predict.py --config config/config.yaml --checkpoint path/to/best_model.pth --image path/to/image.jpg --text "Flood damage in the city center"
```

## ⚙️ Configuration

The model configuration is defined in `config/config.yaml`. Key parameters:

```yaml
model:
  image_model: "resnet50"  # resnet50, efficientnet-b0, vit-base-patch16-224
  text_model: "bert-base-uncased"  # bert-base-uncased, distilbert-base-uncased, roberta-base
  fusion_method: "concatenation"  # concatenation, attention, bilinear
  num_classes: 6
  dropout: 0.3

training:
  epochs: 50
  learning_rate: 0.001
  batch_size: 32
  early_stopping_patience: 10
```

## 📈 Model Architecture

The multimodal model consists of:

1. **Image Encoder**: Pre-trained CNN or Vision Transformer
2. **Text Encoder**: Pre-trained Transformer model
3. **Fusion Module**: Combines image and text features
4. **Classifier**: Final classification head

### Fusion Methods

- **Concatenation**: Simple feature concatenation
- **Attention**: Multi-head attention mechanism
- **Bilinear**: Bilinear pooling for feature interaction

## 📊 Results

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall and per-class accuracy
- **Precision/Recall/F1**: Detailed performance metrics
- **Confusion Matrix**: Visual performance analysis
- **Modality Comparison**: Individual vs. multimodal performance

## 🔧 Advanced Usage

### Custom Model Components

```python
from src.models.multimodal_model import MultimodalModel
from src.data.dataset import create_data_loaders

# Load configuration
config = load_config('config/config.yaml')

# Create model
model = MultimodalModel(config)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(config)
```

### Custom Training Loop

```python
from src.training.trainer import Trainer

trainer = Trainer(model, train_loader, val_loader, config, device='cuda')
best_accuracy = trainer.train()
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code or dataset in your research, please cite the original paper:

```bibtex
@inproceedings{mouzannar2018damage,
  title={Damage Identification in Social Media Posts using Multimodal Deep Learning},
  author={Mouzannar, Hussein and Rizk, Yara and Awad, Mariette},
  booktitle={ISCRAM 2018 Conference Proceedings},
  pages={529--543},
  year={2018},
  organization={Rochester Institute of Technology}
}
```

## 🙏 Acknowledgments

- Original dataset and research by Hussein Mouzannar, Yara Rizk, and Mariette Awad
- Hugging Face for providing pre-trained models
- PyTorch team for the deep learning framework

## 📞 Support

If you have any questions or issues, please:
1. Check the [Issues](https://github.com/your-repo/multimodal-disaster-response/issues) page
2. Create a new issue with detailed information
3. Join our community discussions

---

**Note**: This is a modernized implementation of the original multimodal disaster response model. The original TensorFlow 1.x code has been completely rewritten using modern PyTorch and Transformers libraries for better performance, maintainability, and ease of use.