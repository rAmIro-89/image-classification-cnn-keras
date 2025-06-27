# cnn
Este trabajo practico es de la materia de Procesamiento del Aprendizaje Automatico, esta hecho por Zoe Mlinarevic Medl y Ramiro Ottone.

Para agilizar el trabajo en equipo decidimos hacer el desarrollo en un ide local como vscode y subir los cambios a un gestor de versiones como github.

## Requisitos y Configuración del Entorno

### Versiones de Software Recomendadas

*Python:* 3.9.18  
Versión más estable para Machine Learning con máxima compatibilidad de librerías

### Librerías Principales y Versiones

| Librería | Versión | Propósito |
|----------|---------|-----------|
| *tensorflow* | 2.12.0 | Framework principal para deep learning y CNN |
| *keras* | 2.12.0 | API de alto nivel para redes neuronales (incluida en TF) |
| *numpy* | 1.24.3 | Operaciones numéricas y manejo de arrays |
| *pandas* | 2.0.3 | Análisis y manipulación de datos |
| *matplotlib* | 3.7.2 | Visualización de gráficos y resultados |
| *pillow* | 10.0.0 | Procesamiento y manipulación de imágenes |
| *scikit-learn* | 1.3.0 | Métricas de evaluación y herramientas ML |

### Configuración GPU (Opcional)

Para acelerar el entrenamiento con GPU NVIDIA:

| Componente | Versión | Descripción |
|------------|---------|-------------|
| *CUDA Toolkit* | 11.8 | Plataforma de computación paralela de NVIDIA |
| *cuDNN* | 8.6.0 | Librería de redes neuronales profundas para GPU |

### Instalación del Entorno

*Crear entorno con conda:*
bash
# Crear entorno optimizado
conda create -n ml_stable python=3.9.18 -y
conda activate ml_stable

# Instalar librerías principales
pip install tensorflow==2.12.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install pillow==10.0.0
pip install scikit-learn==1.3.0

# Para soporte GPU (opcional)
conda install cudatoolkit=11.8 cudnn=8.6.0 -c conda-forge -y


*Verificar instalación:*
python
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPUs disponibles: {len(tf.config.list_physical_devices('GPU'))}")


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