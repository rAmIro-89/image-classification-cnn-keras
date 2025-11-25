"""
Example usage of the image classification pipeline for CIFAR-10.

This script demonstrates how to:
1. Load and preprocess CIFAR-10 dataset
2. Create optimized TensorFlow datasets with augmentation
3. Build and train an advanced CNN model
4. Evaluate model performance
5. Visualize training history and predictions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from src.data import (
    load_cifar10,
    normalize_images,
    get_class_names,
    create_tf_datasets
)
from src.models import build_advanced_cnn, build_baseline_cnn
from src.utils import get_training_callbacks
from src.training import train_model, get_optimal_batch_size, get_optimal_epochs
from src.evaluation import (
    evaluate_model,
    plot_training_history,
    predict_and_visualize,
    load_and_evaluate_best_model
)


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("CIFAR-10 Image Classification with CNN")
    print("=" * 70)
    
    # Check GPU availability
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    print(f"\nGPU Available: {gpu_available}")
    
    # Step 1: Load and prepare data
    print("\n[1/5] Loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_cifar10()
    print(f"  Training samples: {len(train_images)}")
    print(f"  Testing samples: {len(test_images)}")
    
    # Step 2: Normalize images
    print("\n[2/5] Normalizing images...")
    train_images, test_images = normalize_images(train_images, test_images)
    
    # Step 3: Create TensorFlow datasets with augmentation
    print("\n[3/5] Creating optimized TensorFlow datasets...")
    batch_size = get_optimal_batch_size()
    print(f"  Batch size: {batch_size} ({'GPU' if gpu_available else 'CPU'} optimized)")
    
    train_dataset, test_dataset = create_tf_datasets(
        train_images, train_labels,
        test_images, test_labels,
        batch_size=batch_size,
        use_augmentation=True
    )
    
    # Step 4: Build model
    print("\n[4/5] Building advanced CNN model...")
    model = build_advanced_cnn()
    print(f"  Total parameters: {model.count_params():,}")
    
    # Display model architecture (abbreviated)
    print("\nModel Architecture Summary:")
    model.summary(print_fn=lambda x: print(x) if 'Total params' in x or 'Trainable' in x else None)
    
    # Step 5: Setup callbacks
    print("\n[5/5] Setting up training callbacks...")
    model_path = 'models/saved/best_model.keras'
    os.makedirs('models/saved', exist_ok=True)
    callbacks = get_training_callbacks(model_path=model_path)
    print(f"  - EarlyStopping (patience=10)")
    print(f"  - ModelCheckpoint (best model saved to {model_path})")
    print(f"  - ReduceLROnPlateau (patience=5)")
    
    # Training
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    epochs = get_optimal_epochs()
    print(f"Training for up to {epochs} epochs...")
    print("(Training may stop early if validation loss stops improving)\n")
    
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    print("\nFinal model performance:")
    final_loss, final_accuracy = evaluate_model(model, test_images, test_labels)
    
    # Load best saved model
    print("\nLoading best saved model...")
    best_model, best_loss, best_accuracy = load_and_evaluate_best_model(
        model_path, test_images, test_labels
    )
    
    # Visualization
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    
    print("\nPlotting training history...")
    plot_training_history(history)
    
    print("\nVisualizing predictions on test samples...")
    class_names = get_class_names()
    predict_and_visualize(
        best_model, test_images, test_labels, class_names, num_images=10
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBest Model Performance:")
    print(f"  Test Accuracy: {best_accuracy*100:.2f}%")
    print(f"  Test Loss: {best_loss:.4f}")
    print(f"\nModel saved to: {model_path}")
    print("\nTraining complete! You can now use the model for predictions.")
    
    # Optional: Show how to use the model for custom predictions
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)
    print("\nTo use the model for predictions on new images:")
    print("""
    from tensorflow import keras
    import numpy as np
    from PIL import Image
    
    # Load the trained model
    model = keras.models.load_model('models/saved/best_model.keras')
    
    # Load and preprocess an image (32x32 pixels)
    img = Image.open('your_image.jpg').resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {np.max(prediction)*100:.2f}%")
    """)


if __name__ == "__main__":
    main()
