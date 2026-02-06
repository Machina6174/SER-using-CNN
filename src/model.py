"""
2D CNN Model for Speech Emotion Recognition.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_ser_model(input_shape: tuple = (128, 130, 1), num_classes: int = 8) -> keras.Model:
    """
    Create a 2D CNN model for Speech Emotion Recognition.
    
    Architecture:
    - 4 Convolutional blocks with BatchNorm and MaxPooling
    - Global Average Pooling
    - Dropout for regularization
    - Dense output layer with softmax
    
    Args:
        input_shape: Input spectrogram shape (height, width, channels)
        num_classes: Number of emotion classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Global Average Pooling (instead of Flatten - reduces overfitting)
        layers.GlobalAveragePooling2D(),
        
        # Dense layers with Dropout (increased for regularization)
        layers.Dropout(0.6),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model: keras.Model, learning_rate: float = 0.001, label_smoothing: float = 0.1) -> keras.Model:
    """
    Compile model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model
        learning_rate: Initial learning rate
        label_smoothing: Label smoothing factor (helps with confused classes)
    
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Use sparse categorical crossentropy (works with integer labels)
    # Note: Label smoothing requires one-hot encoding, so we skip it for simplicity
    # The dropout and class weights provide similar regularization benefits
    loss = 'sparse_categorical_crossentropy'
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_path: str = 'models/best_model.keras', patience: int = 10):
    """
    Get training callbacks.
    
    Args:
        model_path: Path to save best model
        patience: Early stopping patience
    
    Returns:
        List of callbacks
    """
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    # Test model creation
    model = create_ser_model()
    model = compile_model(model)
    model.summary()
