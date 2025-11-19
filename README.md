
# Image Classification with CNNs (CIFAR-10)

This is a professional deep learning project for image classification using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. The repository is structured for clarity, reproducibility, and extensibility, and is intended for my personal data science portfolio.

## Project Overview
This project demonstrates the full workflow for image classification using deep learning. It includes data preprocessing, model development, training, evaluation, and error analysis. The code is modular and ready for experimentation and future improvements.

## Dataset Description
- **Dataset:** CIFAR-10
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images:** 60,000 color images (32x32 pixels)
- **Source:** [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Model Architecture
- Baseline CNN: Multiple convolutional and pooling layers, followed by dense layers.
- Advanced CNN: Includes regularization, data augmentation, and advanced callbacks.
- All models implemented in TensorFlow/Keras.

## Training Process
- Data preprocessing and augmentation
- Model training with early stopping and checkpointing
- Hyperparameter tuning
- Training and validation metrics logged

## Evaluation Results
- Accuracy, loss, and confusion matrix
- Error analysis with misclassified images
- Visualizations of training curves and results

## Environment / Dependencies
- Python 3.9+
- TensorFlow, Keras, NumPy, Pandas, Matplotlib, Pillow, scikit-learn, Jupyter, python-dotenv
- See `requirements.txt` for exact versions

## How to Run the Project
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download CIFAR-10 data or place your images in `data/raw/`
4. Run notebooks in `notebooks/` for EDA, training, and evaluation
5. Use scripts in `src/` for modular pipeline execution

## Project Structure
```
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_cnn_training.ipynb
│   └── 03_evaluation_and_error_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── dataset_loader.py
│   │   ├── preprocessing.py
│   │   └── preprocesar_imagenes.py
│   ├── models/
│   │   ├── cnn_baseline.py
│   │   └── cnn_advanced.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── evaluate.py
│   └── utils/
│       ├── config.py
│       └── callbacks.py
├── models/
│   └── saved/
│       └── mejor_modelo.keras
├── reports/
│   └── figures/
├── docs/
│   ├── architecture.md
│   └── experiments_log.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Future Improvements
- Add more advanced architectures (ResNet, EfficientNet)
- Integrate automated hyperparameter optimization
- Expand error analysis and reporting
- Deploy model as a web service
- Add unit tests and CI/CD pipeline

---
This repository is fully individual and intended for my professional portfolio.

#### EarlyStopping
- **monitor:** 'val_loss' — Supervisa la pérdida de validación para decidir cuándo detener el entrenamiento.
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