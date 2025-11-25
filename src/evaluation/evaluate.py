"""Evaluation utilities for trained CNN models."""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from typing import Tuple, List


def evaluate_model(
    model: keras.Model,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 256
) -> Tuple[float, float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained Keras model
        test_images: Test images
        test_labels: Test labels
        batch_size: Batch size for evaluation
        
    Returns:
        Tuple of (loss, accuracy)
    """
    loss, accuracy = model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=0)
    print(f'\nTest Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    return loss, accuracy


def plot_training_history(history: keras.callbacks.History) -> None:
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: History object from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy Evolution')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss Evolution')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def predict_and_visualize(
    model: keras.Model,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    class_names: List[str],
    num_images: int = 10
) -> None:
    """
    Make predictions and visualize results.
    
    Args:
        model: Trained Keras model
        test_images: Test images
        test_labels: Test labels
        class_names: List of class names
        num_images: Number of images to visualize
    """
    predictions = model.predict(test_images[:num_images])
    
    fig = plt.figure(figsize=(15, 8))
    for i in range(num_images):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[i])
        
        predicted_label = np.argmax(predictions[i])
        true_label = test_labels[i][0]
        confidence = np.max(predictions[i]) * 100
        
        color = 'blue' if predicted_label == true_label else 'red'
        plt.title(f"Pred: {class_names[predicted_label]}\nTrue: {class_names[true_label]}\n{confidence:.1f}%", 
                  color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def load_and_evaluate_best_model(
    model_path: str,
    test_images: np.ndarray,
    test_labels: np.ndarray
) -> Tuple[keras.Model, float, float]:
    """
    Load the best saved model and evaluate it.
    
    Args:
        model_path: Path to saved model
        test_images: Test images
        test_labels: Test labels
        
    Returns:
        Tuple of (model, loss, accuracy)
    """
    model = keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f'\nBest Model Performance:')
    print(f'Loss: {loss:.4f}')
    print(f'Accuracy: {accuracy*100:.2f}%')
    return model, loss, accuracy
