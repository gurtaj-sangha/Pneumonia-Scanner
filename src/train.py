"""
Training script for pneumonia classification model using ResNet-50.
"""
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    ReduceLROnPlateau
)

from data_pipeline import PneumoniaDataPipeline
from model_builder import create_model
from config import MODELS_DIR, LOGS_DIR, PLOTS_DIR

def setup_output_dirs():
    """Create necessary output directories if they don't exist."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main directories
    for dir_path in [MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create run-specific directories
    model_dir = Path(MODELS_DIR) / timestamp
    log_dir = Path(LOGS_DIR) / timestamp
    plot_dir = Path(PLOTS_DIR) / timestamp
    
    for dir_path in [model_dir, log_dir, plot_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return model_dir, log_dir, plot_dir

def create_callbacks(model_dir, log_dir):
    """Create training callbacks for model checkpointing and monitoring."""
    callbacks = [
        # Save best model
        ModelCheckpoint(
            str(model_dir / "best_model.h5"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping if model stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when metrics plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    return callbacks

def plot_training_history(history, plot_dir):
    """Plot and save training metrics."""
    metrics = ['loss', 'accuracy']
    plt.figure(figsize=(12, 4))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 2, i)
        plt.plot(history.history[metric], label='Training')
        plt.plot(history.history[f'val_{metric}'], label='Validation')
        plt.title(f'Model {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'training_history.png')
    plt.close()

def main():
    # Set up directories
    model_dir, log_dir, plot_dir = setup_output_dirs()
    
    # Initialize data pipeline
    pipeline = PneumoniaDataPipeline(
        img_size=(224, 224),
        batch_size=32,
        seed=42
    )
    
    # Get data generators
    train_gen, val_gen, test_gen = pipeline.create_data_generators()
    
    # Calculate class weights for imbalanced dataset
    class_weights = pipeline.calculate_class_weights()
    
    # Create and compile model
    model = create_model(
        input_shape=(224, 224, 3),
        num_classes=3,
        learning_rate=1e-4
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(model_dir, log_dir)
    
    # Train model
    print("\nStarting model training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        class_weight=class_weights,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )
    
    # Plot training history
    plot_training_history(history, plot_dir)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save final model
    model.save(model_dir / "final_model.h5")
    print(f"\nTraining completed. Model saved to {model_dir}")

if __name__ == "__main__":
    main()
