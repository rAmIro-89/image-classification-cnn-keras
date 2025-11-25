
# Image Classification with CNNs (CIFAR-10)

A production-ready deep learning project for image classification using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. This repository demonstrates professional ML engineering practices with modular architecture, comprehensive documentation, and reproducible results.

## Business Problem

**Objective:** Build a robust computer vision system capable of automatically classifying images into 10 distinct categories with high accuracy.

**Real-World Applications:**
- **E-commerce Product Categorization:** Automatically classify product images into relevant categories for inventory management and improved search functionality
- **Content Moderation:** Filter and categorize user-generated images on social media platforms or community forums
- **Autonomous Systems:** Object recognition for self-driving vehicles, drones, or robotics applications
- **Security & Surveillance:** Automated detection and classification of objects in security footage
- **Educational Tools:** Interactive learning applications for teaching computer vision and deep learning concepts

**Technical Challenge:** Handle 32x32 pixel low-resolution images while achieving competitive accuracy (70-85%) across diverse object categories, demonstrating transfer learning potential to real-world image classification tasks.

## Dataset Description
- **Dataset:** CIFAR-10
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images:** 60,000 color images (32x32 pixels)
- **Split:** 50,000 training + 10,000 testing
- **Source:** [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Model Architecture

### Baseline CNN
Simple convolutional architecture with:
- 2 Conv2D blocks (32 and 64 filters)
- MaxPooling layers
- Dense layers for classification

### Advanced CNN (Recommended)
Enhanced architecture with modern deep learning techniques:
- **2 Conv2D blocks** with 64 and 128 filters
- **Batch Normalization** after each convolution for stable training
- **Dropout layers** (0.3 and 0.5) for regularization
- **L2 regularization** in dense layers to prevent overfitting
- **Data Augmentation:** Random flips, rotations, and zoom for improved generalization

## Model Performance

| Model | Test Accuracy | Test Loss | Notes |
|-------|--------------|-----------|-------|
| Baseline CNN | ~70% | ~0.85 | Simple architecture, fast training |
| Advanced CNN | ~80-85% | ~0.45 | Best performance with regularization |

*Note: Results may vary based on hardware (GPU vs CPU) and training duration.*

## How to Run

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd image-classification-cnn-keras

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Start with Example Script
```bash
python example_usage.py
```

### 3. Training from Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Open and run:
# - notebooks/Clasificador_de_imagenes.ipynb (main training notebook)
```

### 4. Using Modular Code (Recommended for Production)
```python
from src.data import load_cifar10, normalize_images, create_tf_datasets
from src.models import build_advanced_cnn
from src.utils import get_training_callbacks
from src.training import train_model
from src.evaluation import evaluate_model, plot_training_history

# Load and prepare data
(train_images, train_labels), (test_images, test_labels) = load_cifar10()
train_images, test_images = normalize_images(train_images, test_images)
train_dataset, test_dataset = create_tf_datasets(
    train_images, train_labels, test_images, test_labels, batch_size=128
)

# Build and train model
model = build_advanced_cnn()
callbacks = get_training_callbacks(model_path='models/saved/best_model.keras')
history = train_model(model, train_dataset, test_dataset, epochs=50, callbacks=callbacks)

# Evaluate
loss, accuracy = evaluate_model(model, test_images, test_labels)
plot_training_history(history)
```

## Project Structure
```
├── data/
│   ├── raw/              # Original CIFAR-10 data (downloaded automatically)
│   └── processed/        # Preprocessed data (if needed)
├── notebooks/
│   └── Clasificador_de_imagenes.ipynb  # Main training notebook
├── src/
│   ├── data/
│   │   ├── dataset_loader.py      # CIFAR-10 loading and preprocessing
│   │   ├── preprocessing.py       # Image preprocessing utilities
│   │   └── preprocesar_imagenes.py  # External image preprocessing
│   ├── models/
│   │   ├── cnn_baseline.py        # Baseline CNN architecture
│   │   └── cnn_advanced.py        # Advanced CNN with regularization
│   ├── training/
│   │   └── train.py               # Training utilities
│   ├── evaluation/
│   │   └── evaluate.py            # Evaluation and visualization
│   └── utils/
│       ├── config.py              # Configuration settings
│       └── callbacks.py           # Keras callbacks (EarlyStopping, etc.)
├── models/
│   └── saved/
│       └── mejor_modelo.keras     # Best trained model
├── reports/
│   └── figures/                   # Training plots and visualizations
├── docs/                          # Additional documentation
├── requirements.txt               # Python dependencies
├── example_usage.py               # Quick start demo script
├── .gitignore
└── README.md
```

## Key Features

### 1. Advanced Training Techniques
- **Data Augmentation:** Random flips, rotations, and zoom applied on-the-fly
- **Callbacks:**
  - **EarlyStopping:** Stops training when validation loss plateaus (patience=10)
  - **ModelCheckpoint:** Saves best model based on validation accuracy
  - **ReduceLROnPlateau:** Reduces learning rate when stuck (factor=0.2, patience=5)
- **Optimized Data Pipeline:** Uses `tf.data.Dataset` with prefetching and parallel processing

### 2. Batch Normalization & Regularization
- **Batch Normalization** after each Conv2D layer for faster convergence
- **Dropout** (0.3 in Conv blocks, 0.5 in Dense layers) to prevent overfitting
- **L2 Regularization** (0.001) in dense layers

### 3. Hardware Optimization
Automatically adapts to available resources:
```python
# GPU detected: batch_size=128, epochs=50
# CPU only: batch_size=32, epochs=25
```

## Training Configuration

### Callbacks Explained

**EarlyStopping:**
- **monitor:** `val_loss` — Monitors validation loss to decide when to stop
- **patience:** 10 — Waits 10 epochs without improvement before stopping
- **restore_best_weights:** True — Restores weights from best epoch

**ModelCheckpoint:**
- **filepath:** `mejor_modelo.keras` — Path where best model is saved
- **monitor:** `val_accuracy` — Saves when validation accuracy improves
- **save_best_only:** True — Only saves improved models
- **mode:** `max` — Maximizes validation accuracy

**ReduceLROnPlateau:**
- **monitor:** `val_loss` — Watches validation loss
- **factor:** 0.2 — Multiplies learning rate by 0.2 when triggered
- **patience:** 5 — Waits 5 epochs before reducing learning rate
- **min_lr:** 0.00001 — Minimum allowed learning rate

### Data Augmentation Pipeline
```python
# Applied during training only
- RandomFlip("horizontal"): Flips images horizontally
- RandomRotation(0.1): Rotates images up to 10% (36 degrees)
- RandomZoom(0.1): Zooms from 90% to 110%
```

## Lessons Learned

### Importance of Preprocessing
When classifying external images (not from CIFAR-10), initial performance was poor due to:
- **Training data:** 32x32 images where objects fill most of the space
- **Real-world data:** High-resolution images where objects are small parts of the scene

**Solution:** Implemented centered cropping before resizing (`preprocesar_imagenes.py`), significantly improving real-world accuracy.

### Model Generalization
The model acts like a "student" trained on specific material. For real-world applications, input data must maintain the same structure and quality as training data.

### Callback Automation
Keras callbacks were crucial for:
- **Automating** decisions during training
- **Preventing overfitting** without manual intervention
- **Optimizing** training time
- **Ensuring** best model is saved

This demonstrates that successful training depends not only on architecture but also on intelligent training strategies.

## Future Improvements
- Implement advanced architectures (ResNet, EfficientNet, Vision Transformers)
- Add automated hyperparameter optimization (Optuna, Keras Tuner)
- Expand error analysis with confusion matrices and misclassification reports
- Deploy model as REST API or web service
- Add unit tests and CI/CD pipeline for automated training
- Experiment with transfer learning from ImageNet

## Technologies Used
- **Python 3.9+**
- **TensorFlow 2.10+** & Keras for deep learning
- **NumPy** & **Pandas** for data manipulation
- **Matplotlib** for visualization
- **scikit-learn** for evaluation metrics
- **Jupyter** for interactive notebooks

---

**Author:** Portfolio project for Canadian tech market  
**License:** MIT  
**Status:** Production-ready for demonstration and extension
- **patience:** 10 — Número de épocas a esperar sin mejora antes de detener el entrenamiento.
- **restore_best_weights:** True — Restaura los pesos del modelo al mejor valor encontrado durante el entrenamiento.

#### ModelCheckpoint
- **filepath:** 'mejor_modelo.keras' — Ruta y nombre del archivo donde se guarda el modelo.
- **monitor:** 'val_accuracy' — Supervisa la precisión de validación para guardar el mejor modelo.
- **save_best_only:** True — Solo guarda el modelo si mejora la métrica monitoreada.
- **mode:** 'max' — Busca maximizar la métrica monitoreada (en este caso, la precisión).

#### ReduceLROnPlateau
- **monitor:** 'val_loss' — Supervisa la pérdida de validación para reducir el learning rate si no mejora.
- **factor:** 0.2 — Multiplica el learning rate actual por este factor cuando se activa.
- **patience:** 5 — Número de épocas a esperar sin mejora antes de reducir el learning rate.
- **min_lr:** 0.00001 — Valor mínimo permitido para el learning rate.

### Instalación del Entorno

*Crear entorno con conda:*
```bash
# Crear entorno optimizado
conda create -n ml_stable python=3.9.21 -y
conda activate ml_stable
```

# Instalar librerías principales
pip install tensorflow==2.10.0
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install matplotlib==3.5.2
pip install pillow==9.2.0
pip install scikit-learn==1.1.3

# Para soporte GPU (opcional)
conda install cudatoolkit=11.8 cudnn=8.6.0 -c conda-forge -y

*Verificar instalación:*
```python
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPUs disponibles: {len(tf.config.list_physical_devices('GPU'))}")
```


### Dataset

- *CIFAR-10:* 60,000 imágenes en color de 32x32 píxeles
- *Clases:* 10 categorías (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- *División:* 50,000 imágenes de entrenamiento + 10,000 de validación

# Arquitectura del modelo CNN

Este proyecto utiliza una red neuronal convolucional (CNN) para clasificar imágenes del dataset CIFAR-10.  
A continuación se describen los componentes principales, sus definiciones y cómo pueden mejorarse:

## Estructura Mejorada de la Red

### Capas Convolucionales Optimizadas

*Configuración implementada:*
python
# Primer bloque convolucional
modelo.add(Conv2D(64, (3,3), padding="same", activation='relu', input_shape=(32,32,3)))
modelo.add(BatchNormalization())
modelo.add(Conv2D(64, (3,3), padding="same", activation='relu'))  # Capa extra agregada
modelo.add(BatchNormalization()) 
modelo.add(MaxPool2D(2,2))
modelo.add(Dropout(0.3))

# Segundo bloque convolucional
modelo.add(Conv2D(128, (3,3), padding="same", activation='relu'))  # Filtros aumentados de 64 a 128
modelo.add(BatchNormalization())
modelo.add(Conv2D(128, (3,3), padding="same", activation='relu'))  # Capa extra agregada
modelo.add(BatchNormalization())
modelo.add(MaxPool2D(2,2))
modelo.add(Dropout(0.3))


*Justificación de las mejoras implementadas:*

1. *Capas Convolucionales Adicionales*
   - *¿Qué agregamos?* Una segunda capa Conv2D en cada bloque
   - *¿Por qué?* Permite al modelo aprender características más complejas antes de reducir dimensiones
   - *Beneficio:* Mejor extracción de patrones detallados sin perder información espacial

2. *Aumento de Filtros (64 → 128)*
   - *¿Qué agregamos?* Incremento progresivo en el número de filtros
   - *¿Por qué?* A medida que profundizamos, necesitamos más filtros para capturar patrones más abstractos
   - *Beneficio:* Mayor capacidad para detectar características complejas

3. *BatchNormalization después de cada Conv2D*
   - *¿Qué agregamos?* Normalización de lotes en cada capa convolucional
   - *¿Por qué?* Estabiliza el entrenamiento y acelera la convergencia
   - *Beneficio:* Permite usar learning rates más altos y reduce la sensibilidad a la inicialización

4. *Dropout aumentado (0.25 → 0.3)*
   - *¿Qué agregamos?* Mayor regularización
   - *¿Por qué?* Con más parámetros, necesitamos más regularización para evitar overfitting
   - *Beneficio:* Mejor generalización del modelo

## Callbacks de Keras Implementados

### 1. EarlyStopping
python
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

- *Propósito:* Detiene el entrenamiento automáticamente cuando el modelo deja de mejorar
- *Parámetros:*
  - monitor='val_loss': Supervisa la pérdida de validación
  - patience=10: Espera 10 épocas sin mejora antes de parar
  - restore_best_weights=True: Restaura los mejores pesos encontrados
- *Beneficio:* Previene overfitting y ahorra tiempo de entrenamiento

### 2. ModelCheckpoint
python
checkpoint = ModelCheckpoint('mejor_modelo.keras', monitor='val_accuracy', save_best_only=True, mode='max')

- *Propósito:* Guarda automáticamente el mejor modelo durante el entrenamiento
- *Parámetros:*
  - 'mejor_modelo.keras': Nombre del archivo donde se guarda
  - monitor='val_accuracy': Supervisa la precisión de validación
  - save_best_only=True: Solo guarda cuando encuentra un mejor modelo
  - mode='max': Busca maximizar la métrica (accuracy)
- *Beneficio:* Garantiza que no perdemos el mejor modelo aunque el entrenamiento continúe

### 3. ReduceLROnPlateau
python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

- *Propósito:* Reduce automáticamente el learning rate cuando el entrenamiento se estanca
- *Parámetros:*
  - monitor='val_loss': Supervisa la pérdida de validación
  - factor=0.2: Multiplica el LR por 0.2 (lo reduce a 1/5)
  - patience=5: Espera 5 épocas sin mejora antes de reducir
  - min_lr=0.00001: Learning rate mínimo permitido
- *Beneficio:* Permite al modelo hacer ajustes más finos cuando se acerca al óptimo

## Data Augmentation Implementado

python
data_augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


*Técnicas aplicadas:*
- *RandomFlip("horizontal"):* Voltea imágenes horizontalmente (útil para objetos simétricos)
- *RandomRotation(0.1):* Rota imágenes hasta 10% (36 grados)
- *RandomZoom(0.1):* Aplica zoom del 90% al 110%

*Beneficios:*
- Aumenta artificialmente el dataset
- Mejora la generalización del modelo
- Reduce overfitting
- Se ejecuta en GPU para mayor eficiencia

## Optimizaciones de Pipeline de Datos

### tf.data.Dataset Optimizado
python
# Dataset de entrenamiento con optimizaciones
train_dataset = tf.data.Dataset.from_tensor_slices((imagenes_entrenamiento, etiquetas_entrenamiento))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
train_dataset = train_dataset.map(lambda x, y: (data_augmentation_layers(x, training=True), y), num_parallel_calls=autotune)
train_dataset = train_dataset.prefetch(buffer_size=autotune)


*Optimizaciones clave:*
- *shuffle():* Mezcla los datos para evitar sesgos de orden
- *batch():* Procesa datos en lotes para eficiencia
- *map() con num_parallel_calls:* Aplica transformaciones en paralelo
- *prefetch():* Prepara el siguiente lote mientras se procesa el actual
- *autotune:* TensorFlow optimiza automáticamente los parámetros

## Configuración Adaptativa

El código se adapta automáticamente según los recursos disponibles:

python
if len(tf.config.list_physical_devices('GPU')) > 0:
    batch_size = 128  # Para GPU
    epochs = 50
else:
    batch_size = 32   # Para CPU
    epochs = 25


### Otras técnicas implementadas

- *Regularización L2:* En la capa densa para penalizar pesos grandes
- *Softmax:* En la capa de salida para clasificación multiclase
- *Adam Optimizer:* Optimizador adaptativo para convergencia más rápida
- *sparse_categorical_crossentropy:* Función de pérdida optimizada para clasificación

# Implemento
Aumentar la Capacidad del Modelo (Más Filtros y Neuronas)
Un modelo con más filtros y neuronas puede aprender patrones más complejos y detallados de las imágenes, lo que a menudo conduce a una mayor precisión.

## Lecciones Aprendidas

### Importancia del Preprocesamiento
Durante el desarrollo descubrimos que la *preparación de datos* es tan crucial como la arquitectura del modelo. Al intentar clasificar imágenes externas (no del dataset CIFAR-10), el modelo inicialmente tenía bajo rendimiento debido a la diferencia entre:

- *Datos de entrenamiento:* Imágenes de 32x32 donde el objeto ocupa casi todo el espacio
- *Datos reales:* Imágenes de alta resolución donde el objeto es una pequeña parte de la escena

*Solución implementada:* Script de preprocesamiento que realiza recorte centrado antes de redimensionar, mejorando significativamente la precisión en imágenes del mundo real.

### Generalización del Modelo
El modelo funciona como un "estudiante" que solo conoce un tipo específico de material de estudio. Para que funcione en aplicaciones reales, es fundamental que los datos de entrada mantengan la misma estructura y calidad que los datos de entrenamiento.

### Importancia de los Callbacks
Los callbacks de Keras fueron fundamentales para:
- *Automatizar* decisiones durante el entrenamiento
- *Prevenir overfitting* sin intervención manual
- *Optimizar* el tiempo de entrenamiento
- *Garantizar* que se guarde el mejor modelo

Esto demuestra que un buen entrenamiento no solo depende de la arquitectura, sino también de una estrategia de entrenamiento inteligente.

################################################

Mejoras para aumentar la precisión
Se aumentó el número de epochs a 100 y el batch size a 256.
Se agregaron capas adicionales (más filtros, batch normalization, dropout) para mejorar la generalización.
Se implementaron callbacks avanzados: EarlyStopping, ModelCheckpoint y ReduceLROnPlateau.
Estos cambios se realizaron para mejorar la precisión y reducir el overfitting.
Predicción de imágenes externas
Se implementó la función predecir_imagen_externa, que permite cargar y clasificar imágenes externas (fotos, dibujos, etc.) en una de las 10 clases del modelo entrenado.

Funcionamiento:

Carga la imagen y la redimensiona a 32x32 píxeles.
Convierte la imagen a un array NumPy y la normaliza.
Agrega una dimensión extra para simular un batch.
Realiza la predicción con el modelo entrenado.
Muestra la imagen junto con la clase predicha y el porcentaje de confianza.
Uso:
La función se utiliza en un bucle que recorre todas las imágenes de una carpeta (pics_procesadas) y muestra la predicción para cada una.