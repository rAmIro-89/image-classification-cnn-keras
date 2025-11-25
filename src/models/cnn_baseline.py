"""Baseline CNN model for CIFAR-10 image classification."""

import tensorflow as tf
from tensorflow import keras


def build_baseline_cnn(input_shape: tuple = (32, 32, 3), num_classes: int = 10) -> keras.Model:
    """
    Build a baseline CNN model for CIFAR-10 classification.
    
    Architecture:
        - Conv2D (32 filters) + MaxPooling
        - Conv2D (64 filters) + MaxPooling
        - Flatten + Dense (64 neurons)
        - Output Dense (10 classes)
    
    Args:
        input_shape: Shape of input images (default: 32x32x3 for CIFAR-10)
        num_classes: Number of output classes (default: 10)
        
    Returns:
        Compiled Keras model
    """
    model = keras.models.Sequential([
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPool2D(2, 2),
        
        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D(2, 2),
        
        # Fully connected layers
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
