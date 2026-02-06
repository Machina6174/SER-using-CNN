"""
Training script for Speech Emotion Recognition.

Uses pre-split data (train/val/test) from preprocessing.
No additional splitting needed - data is already split by actor.
"""

import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import create_ser_model, compile_model, get_callbacks
from src.preprocessing import IDX_TO_EMOTION


def load_split_data(data_dir: str = 'data/processed'):
    """
    Load pre-split data (already split by actor in preprocessing).
    
    Returns:
        dict: {'train': (X, y, meta), 'val': (X, y, meta), 'test': (X, y, meta)}
    """
    data = {}
    
    for split in ['train', 'val', 'test']:
        X = np.load(os.path.join(data_dir, f'X_{split}.npy'))
        y = np.load(os.path.join(data_dir, f'y_{split}.npy'))
        
        with open(os.path.join(data_dir, f'metadata_{split}.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        data[split] = (X, y, metadata)
        print(f"Loaded {split}: X={X.shape}, y={y.shape}")
    
    return data


def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, use_class_weights=True):
    """
    Train the CNN model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        use_class_weights: Whether to use class weights for imbalanced classes
    
    Returns:
        tuple: (model, history)
    """
    # Create and compile model
    model = create_ser_model(input_shape=X_train.shape[1:])
    model = compile_model(model)
    
    print("\nModel Summary:")
    model.summary()
    
    # Calculate class weights if requested
    class_weight = None
    if use_class_weights:
        from collections import Counter
        class_counts = Counter(y_train)
        total = len(y_train)
        n_classes = len(class_counts)
        
        # Compute balanced weights: higher weight for underrepresented classes
        class_weight = {}
        for cls, count in class_counts.items():
            # Weight = total / (n_classes * count) gives balanced weights
            class_weight[cls] = total / (n_classes * count)
        
        print("\nClass weights (to handle imbalance):")
        for cls, weight in sorted(class_weight.items()):
            print(f"  {IDX_TO_EMOTION[cls]:>10}: {weight:.3f}")
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, X_test, y_test, meta_test=None):
    """
    Evaluate model on test set.
    
    IMPORTANT: Test set contains ONLY original samples from held-out actors.
    No augmented data. No data leakage.
    
    Returns:
        dict: Evaluation results
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # Classification report
    target_names = [IDX_TO_EMOTION[i] for i in range(8)]
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS (Held-out actors, no augmentation)")
    print("="*50)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Gender bias analysis
    if meta_test is not None:
        male_mask = np.array([m['gender'] == 'male' for m in meta_test])
        female_mask = ~male_mask
        
        if np.sum(male_mask) > 0 and np.sum(female_mask) > 0:
            male_acc = np.mean(y_pred[male_mask] == y_test[male_mask])
            female_acc = np.mean(y_pred[female_mask] == y_test[female_mask])
            
            print("\nGender Bias Analysis:")
            print(f"  Male Accuracy:   {male_acc:.4f}")
            print(f"  Female Accuracy: {female_acc:.4f}")
            print(f"  Difference:      {abs(male_acc - female_acc):.4f}")
    
    # Confidence analysis
    max_probs = np.max(y_pred_proba, axis=1)
    correct_mask = y_pred == y_test
    
    print("\nConfidence Analysis:")
    print(f"  Correct predictions - Mean conf: {np.mean(max_probs[correct_mask])*100:.1f}%")
    if np.sum(~correct_mask) > 0:
        print(f"  Wrong predictions   - Mean conf: {np.mean(max_probs[~correct_mask])*100:.1f}%")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'report': report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


if __name__ == "__main__":
    # Load pre-split data (split by actor in preprocessing)
    print("Loading pre-split data...")
    print("(Data was split by ACTOR to prevent speaker leakage)")
    print()
    
    data = load_split_data()
    
    X_train, y_train, meta_train = data['train']
    X_val, y_val, meta_val = data['val']
    X_test, y_test, meta_test = data['test']
    
    print(f"\nData summary:")
    print(f"  Train: {len(X_train)} samples (with augmentation)")
    print(f"  Val:   {len(X_val)} samples (original only)")
    print(f"  Test:  {len(X_test)} samples (original only)")
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluate on held-out test actors
    print("\n" + "="*60)
    print("TESTING ON HELD-OUT ACTORS (23, 24)")
    print("These voices were NEVER heard during training!")
    print("="*60)
    
    results = evaluate_model(model, X_test, y_test, meta_test)
    
    print("\nTraining complete!")
