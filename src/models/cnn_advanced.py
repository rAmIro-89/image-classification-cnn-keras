"""Advanced CNN model with BatchNormalization and Dropout for CIFAR-10."""

import tensorflow as tf
from tensorflow import keras


def build_advanced_cnn(input_shape: tuple = (32, 32, 3), num_classes: int = 10) -> keras.Model:
    """
    Build an advanced CNN model with regularization and normalization.
    
    Architecture:
        - Conv2D (64 filters) + BatchNorm + Conv2D (64) + BatchNorm + MaxPool + Dropout
        - Conv2D (128 filters) + BatchNorm + Conv2D (128) + BatchNorm + MaxPool + Dropout
        - Flatten + Dense (128 neurons with L2 regularization) + Dropout
        - Output Dense (10 classes)
    
    Features:
        - Batch normalization for faster convergence
        - Dropout for regularization
        - L2 regularization in dense layers
        - Increased model capacity
    
    Args:
        input_shape: Shape of input images (default: 32x32x3 for CIFAR-10)
        num_classes: Number of output classes (default: 10)
        
    Returns:
        Compiled Keras model
    """
    model = keras.models.Sequential([
        # First convolutional block
        keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Dropout(0.3),
        
        # Second convolutional block
        keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Dropout(0.3),
        
        # Fully connected layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
