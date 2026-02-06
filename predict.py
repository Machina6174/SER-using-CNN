"""
Live Inference Script for Speech Emotion Recognition.

Usage:
    python predict.py path/to/audio.wav
    
Output:
    Predicted emotion: Happy (87.3% confidence)
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import (
    load_audio, trim_silence, pad_or_truncate,
    extract_mel_spectrogram, pad_spectrogram,
    SAMPLE_RATE, DURATION, IDX_TO_EMOTION
)


def load_model(model_path: str = 'models/best_model.keras'):
    """Load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first by running the notebook or train.py"
        )
    return tf.keras.models.load_model(model_path)


def preprocess_audio(filepath: str) -> np.ndarray:
    """
    Preprocess a single audio file for prediction.
    
    Args:
        filepath: Path to .wav file
    
    Returns:
        Preprocessed spectrogram ready for model input
    """
    # Load audio
    audio, sr = load_audio(filepath, sr=SAMPLE_RATE)
    
    # Trim silence
    audio = trim_silence(audio)
    
    # Pad/truncate to fixed duration
    target_samples = int(DURATION * sr)
    audio = pad_or_truncate(audio, target_samples)
    
    # Extract Mel-spectrogram
    mel_spec = extract_mel_spectrogram(audio, sr)
    
    # Pad to uniform shape
    mel_spec = pad_spectrogram(mel_spec)
    
    # Add batch and channel dimensions
    mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
    
    return mel_spec


def predict_emotion(model, filepath: str) -> tuple:
    """
    Predict emotion from audio file.
    
    Args:
        model: Loaded Keras model
        filepath: Path to .wav file
    
    Returns:
        tuple: (predicted_emotion, confidence_percentage)
    """
    # Preprocess
    spectrogram = preprocess_audio(filepath)
    
    # Predict
    predictions = model.predict(spectrogram, verbose=0)
    
    # Get top prediction
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    
    predicted_emotion = IDX_TO_EMOTION[predicted_idx]
    
    return predicted_emotion, confidence, predictions[0]


def main():
    parser = argparse.ArgumentParser(
        description='Predict emotion from speech audio file'
    )
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to the .wav audio file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.keras',
        help='Path to the trained model (default: models/best_model.keras)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show all emotion probabilities'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    if not args.audio_file.lower().endswith('.wav'):
        print("⚠️  Warning: File doesn't have .wav extension. Attempting to process anyway...")
    
    # Load model
    print(f"Loading model from {args.model}...")
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Predict
    print(f"Processing: {args.audio_file}")
    emotion, confidence, all_probs = predict_emotion(model, args.audio_file)
    
    # Output
    print("\n" + "="*50)
    print(f"🎯 Predicted emotion: {emotion.upper()} ({confidence:.1f}% confidence)")
    print("="*50)
    
    if args.verbose:
        print("\nAll emotion probabilities:")
        sorted_probs = sorted(
            [(IDX_TO_EMOTION[i], all_probs[i] * 100) for i in range(8)],
            key=lambda x: x[1],
            reverse=True
        )
        for emotion_name, prob in sorted_probs:
            bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
            print(f"  {emotion_name:12s} {bar} {prob:5.1f}%")


if __name__ == "__main__":
    main()
