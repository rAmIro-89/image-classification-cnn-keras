"""Keras callbacks for model training optimization."""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import List


def get_training_callbacks(
    model_path: str = 'mejor_modelo.keras',
    early_stop_patience: int = 10,
    reduce_lr_patience: int = 5,
    min_lr: float = 0.00001
) -> List:
    """
    Create a list of Keras callbacks for training optimization.
    
    Callbacks included:
        - EarlyStopping: Stops training when validation loss stops improving
        - ModelCheckpoint: Saves the best model based on validation accuracy
        - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus
    
    Args:
        model_path: Path to save the best model
        early_stop_patience: Number of epochs with no improvement before stopping
        reduce_lr_patience: Number of epochs with no improvement before reducing LR
        min_lr: Minimum learning rate
        
    Returns:
        List of Keras callbacks
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=early_stop_patience,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1
    )
    
    return [early_stop, checkpoint, reduce_lr]
