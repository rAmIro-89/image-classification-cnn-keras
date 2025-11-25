"""Training utilities for CNN models on CIFAR-10."""

import tensorflow as tf
from tensorflow import keras
from typing import Optional


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = 100,
    callbacks: Optional[list] = None
) -> keras.callbacks.History:
    """
    Train a CNN model on CIFAR-10 dataset.
    
    Args:
        model: Keras model to train
        train_dataset: Training dataset (tf.data.Dataset)
        val_dataset: Validation dataset (tf.data.Dataset)
        epochs: Number of training epochs
        callbacks: List of Keras callbacks
        
    Returns:
        History object containing training metrics
    """
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks if callbacks else [],
        verbose=1
    )
    
    return history


def get_optimal_batch_size() -> int:
    """
    Determine optimal batch size based on available hardware.
    
    Returns:
        Recommended batch size (128 for GPU, 32 for CPU)
    """
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    return 128 if gpu_available else 32


def get_optimal_epochs() -> int:
    """
    Determine optimal number of epochs based on available hardware.
    
    Returns:
        Recommended epochs (50 for GPU, 25 for CPU)
    """
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    return 50 if gpu_available else 25
