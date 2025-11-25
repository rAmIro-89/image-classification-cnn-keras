"""Dataset loading and preprocessing utilities for CIFAR-10 image classification."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple


def load_cifar10() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load CIFAR-10 dataset from Keras datasets.
    
    Returns:
        Tuple containing:
            - Training data (images, labels)
            - Testing data (images, labels)
    """
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    return (train_images, train_labels), (test_images, test_labels)


def normalize_images(train_images: np.ndarray, test_images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize image pixel values from [0, 255] to [0, 1].
    
    Args:
        train_images: Training images array
        test_images: Testing images array
        
    Returns:
        Tuple of normalized (train_images, test_images)
    """
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, test_images


def get_class_names() -> list:
    """
    Get CIFAR-10 class names.
    
    Returns:
        List of 10 class names
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']


def create_data_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Create data augmentation pipeline for training.
    
    Returns:
        Sequential model with augmentation layers
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    return data_augmentation


def create_tf_datasets(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 256,
    use_augmentation: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create optimized TensorFlow datasets with optional data augmentation.
    
    Args:
        train_images: Training images
        train_labels: Training labels
        test_images: Testing images
        test_labels: Testing labels
        batch_size: Batch size for training
        use_augmentation: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    autotune = tf.data.AUTOTUNE
    
    # Training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    if use_augmentation:
        augmentation_layers = create_data_augmentation_pipeline()
        train_dataset = train_dataset.map(
            lambda x, y: (augmentation_layers(x, training=True), y),
            num_parallel_calls=autotune
        )
    
    train_dataset = train_dataset.prefetch(buffer_size=autotune)
    
    # Test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=autotune)
    
    return train_dataset, test_dataset
