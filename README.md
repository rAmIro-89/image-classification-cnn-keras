# Image Classification with CNN (Keras/TensorFlow)

End-to-end image classification project using Convolutional Neural Networks (CNN) with Keras/TensorFlow. This project provides a complete, professional structure for building, training, and evaluating deep learning models for image classification tasks.

## 🎯 Project Overview

This repository contains a fully-structured machine learning project for image classification using CNNs. It includes:
- Modular code architecture for data processing, model building, training, and evaluation
- Organized directory structure following best practices
- Comprehensive documentation and examples
- Ready-to-use training and evaluation pipelines

## 📁 Project Structure

```
image-classification-cnn-keras/
├── data/
│   ├── raw/                    # Original, immutable datasets
│   └── processed/              # Cleaned and preprocessed data
├── notebooks/                  # Jupyter notebooks for EDA and experiments
├── src/
│   ├── data/                   # Data loading and preprocessing modules
│   ├── models/                 # Model architectures and definitions
│   ├── training/               # Training pipelines and callbacks
│   ├── evaluation/             # Evaluation metrics and visualization
│   └── utils/                  # Utility functions and helpers
├── models/
│   └── saved/                  # Trained model files
├── reports/
│   └── figures/                # Generated plots and visualizations
├── docs/                       # Project documentation
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE                     # Project license
```

## 📊 Dataset

This project is designed to work with image classification datasets. You can use popular datasets such as:

- **CIFAR-10/CIFAR-100**: 60,000 32x32 color images in 10/100 classes
- **MNIST/Fashion-MNIST**: 70,000 28x28 grayscale images
- **ImageNet subset**: High-resolution images across multiple categories
- **Custom datasets**: Any image classification dataset organized in folders by class

### Dataset Setup

1. Download your chosen dataset
2. Place raw data in `data/raw/`
3. Organize images in subdirectories by class:
   ```
   data/raw/
   ├── class1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── class2/
   │   └── ...
   └── ...
   ```
4. Run preprocessing scripts to generate processed data in `data/processed/`

## 🧠 Model Architecture

The project supports various CNN architectures:

### Basic CNN
- Multiple convolutional layers with ReLU activation
- Max pooling for spatial dimension reduction
- Dropout layers for regularization
- Dense layers for classification

### Advanced Architectures
- **VGG-style**: Deep networks with small 3x3 filters
- **ResNet-style**: Residual connections for deeper networks
- **Custom architectures**: Easily extensible for your own designs

### Model Features
- Batch normalization for training stability
- Data augmentation (rotation, flip, zoom, shift)
- Early stopping and learning rate scheduling
- Model checkpointing to save best weights

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rAmIro-89/image-classification-cnn-keras.git
   cd image-classification-cnn-keras
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎓 Training

### Basic Training

```python
# Example training script (to be implemented in src/training/)
from src.data.data_loader import load_data
from src.models.cnn_model import build_model
from src.training.trainer import train_model

# Load and preprocess data
train_data, val_data = load_data('data/processed/')

# Build model
model = build_model(input_shape=(224, 224, 3), num_classes=10)

# Train model
history = train_model(
    model=model,
    train_data=train_data,
    val_data=val_data,
    epochs=50,
    batch_size=32
)
```

### Training Configuration

Key hyperparameters:
- **Learning rate**: 0.001 (with decay)
- **Batch size**: 32-128 (depending on GPU memory)
- **Optimizer**: Adam, SGD, or RMSprop
- **Loss function**: Categorical crossentropy
- **Epochs**: 50-100 with early stopping

### Training Features
- Automatic checkpointing of best models
- TensorBoard logging for visualization
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping to prevent overfitting

## 📈 Evaluation

### Metrics

The project calculates and reports:
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class and macro/micro averages
- **Confusion Matrix**: Visualization of predictions vs. actual labels
- **ROC Curves**: For binary and multi-class classification

### Evaluation Example

```python
from src.evaluation.evaluator import evaluate_model

# Load test data and trained model
test_data = load_data('data/processed/test/')
model = load_model('models/saved/best_model.h5')

# Evaluate
results = evaluate_model(model, test_data)
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test Loss: {results['loss']:.4f}")
```

### Visualization

Generate evaluation plots:
- Training/validation accuracy and loss curves
- Confusion matrix heatmap
- Sample predictions with confidence scores
- Misclassified examples analysis

## 🛠️ How to Run

### 1. Data Preparation
```bash
# Place your dataset in data/raw/
# Run preprocessing (script to be implemented)
python src/data/preprocess.py --input data/raw/ --output data/processed/
```

### 2. Model Training
```bash
# Train the model (script to be implemented)
python src/training/train.py --data data/processed/ --epochs 50 --batch-size 32
```

### 3. Model Evaluation
```bash
# Evaluate on test set (script to be implemented)
python src/evaluation/evaluate.py --model models/saved/best_model.h5 --data data/processed/test/
```

### 4. Make Predictions
```bash
# Predict on new images (script to be implemented)
python src/utils/predict.py --model models/saved/best_model.h5 --image path/to/image.jpg
```

## 📓 Notebooks

Explore the `notebooks/` directory for:
- Data exploration and visualization
- Model architecture experiments
- Training process analysis
- Results interpretation

## 🔧 Development

### Adding New Features

1. **New data preprocessing**: Add modules to `src/data/`
2. **New model architecture**: Define in `src/models/`
3. **Custom training logic**: Extend `src/training/`
4. **Additional metrics**: Implement in `src/evaluation/`

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Document functions with docstrings
- Keep functions focused and modular

## 📚 Dependencies

Core dependencies:
- TensorFlow/Keras: Deep learning framework
- NumPy: Numerical computing
- Pandas: Data manipulation
- Matplotlib/Seaborn: Visualization
- OpenCV/Pillow: Image processing
- scikit-learn: Metrics and utilities

See `requirements.txt` for complete list and versions.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow and Keras teams for excellent frameworks
- Open-source ML community for inspiration and best practices
- Dataset providers for making data publicly available

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a template/starter project. Actual implementation of training scripts, model architectures, and data processing pipelines should be added based on your specific use case and dataset.
