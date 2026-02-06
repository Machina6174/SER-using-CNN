"""
Preprocessing module for Speech Emotion Recognition.
Handles audio loading, cleaning, Mel-spectrogram extraction, and augmentation.

METHODOLOGY:
- Stratified random split (maintains class balance)
- Augmentation applied ONLY to training data (no data leakage)
- Test/Val sets contain only original samples

NOTE: This uses sample-level splits, not speaker-level splits.
For truly speaker-independent evaluation, use actor-based splits.
"""

import os
import numpy as np
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split


# RAVDESS emotion labels mapping
# Filename format: XX-XX-XX-XX-XX-XX-XX.wav
# Position 3 (0-indexed: 2) is the emotion
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_MAP.values())}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 3.0  # seconds
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SHAPE = (128, 130)  # (n_mels, time_frames) - uniform shape for all spectrograms


def parse_filename(filepath: str) -> dict:
    """
    Parse RAVDESS filename to extract metadata.
    
    Filename format: modality-vocal-emotion-intensity-statement-repetition-actor.wav
    """
    filename = os.path.basename(filepath)
    parts = filename.replace('.wav', '').split('-')
    
    if len(parts) != 7:
        raise ValueError(f"Invalid RAVDESS filename format: {filename}")
    
    return {
        'modality': parts[0],      # 01 = audio-only
        'vocal_channel': parts[1], # 01 = speech
        'emotion': EMOTION_MAP.get(parts[2], 'unknown'),
        'emotion_idx': EMOTION_TO_IDX.get(EMOTION_MAP.get(parts[2], 'unknown'), -1),
        'intensity': parts[3],     # 01 = normal, 02 = strong
        'statement': parts[4],     # 01 or 02
        'repetition': parts[5],    # 01 or 02
        'actor': int(parts[6]),    # 01-24 (odd = male, even = female)
        'gender': 'male' if int(parts[6]) % 2 == 1 else 'female',
        'filepath': filepath
    }


def load_audio(filepath: str, sr: int = SAMPLE_RATE) -> tuple:
    """Load audio file and resample if necessary."""
    audio, sr = librosa.load(filepath, sr=sr)
    return audio, sr


def trim_silence(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
    """Remove silence from the beginning and end of audio."""
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad audio with zeros or truncate to match target length."""
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        padding = target_length - len(audio)
        return np.pad(audio, (0, padding), mode='constant')
    return audio


def extract_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convert audio to Log-Mel Spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


def pad_spectrogram(spectrogram: np.ndarray, target_shape: tuple = TARGET_SHAPE) -> np.ndarray:
    """Pad or truncate spectrogram to uniform shape."""
    n_mels, time_frames = spectrogram.shape
    target_mels, target_frames = target_shape
    
    if time_frames > target_frames:
        spectrogram = spectrogram[:, :target_frames]
    elif time_frames < target_frames:
        padding = target_frames - time_frames
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')
    
    if n_mels != target_mels:
        raise ValueError(f"Mel dimension mismatch: {n_mels} vs {target_mels}")
    
    return spectrogram


def create_augmenter(p: float = 0.5) -> Compose:
    """Create audio augmentation pipeline."""
    return Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p),
        PitchShift(min_semitones=-4, max_semitones=4, p=p),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=p),
    ])


def augment_audio(audio: np.ndarray, sr: int = SAMPLE_RATE, augmenter: Compose = None) -> np.ndarray:
    """Apply random augmentations to audio."""
    if augmenter is None:
        augmenter = create_augmenter()
    return augmenter(samples=audio, sample_rate=sr)


def process_single_file(filepath: str, augment: bool = False, augmenter: Compose = None) -> tuple:
    """Process a single audio file through the full pipeline."""
    audio, sr = load_audio(filepath)
    audio = trim_silence(audio)
    target_samples = int(DURATION * sr)
    audio = pad_or_truncate(audio, target_samples)
    
    if augment:
        audio = augment_audio(audio, sr, augmenter)
    
    mel_spec = extract_mel_spectrogram(audio, sr)
    mel_spec = pad_spectrogram(mel_spec)
    metadata = parse_filename(filepath)
    
    return mel_spec, metadata


def process_dataset_stratified(data_dir: str, output_dir: str, n_augmentations: int = 2,
                                test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42):
    """
    Process RAVDESS dataset with stratified random splits.
    
    IMPORTANT: 
    - Split FIRST, then augment ONLY training data
    - Test/Val sets contain only original samples (no augmentation)
    - This prevents data leakage
    
    Args:
        data_dir: Path to raw audio files
        output_dir: Path to save processed spectrograms
        n_augmentations: Number of augmented copies per training file
        test_size: Proportion of test set
        val_size: Proportion of validation set
        random_state: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                audio_files.append(filepath)
                meta = parse_filename(filepath)
                labels.append(meta['emotion_idx'])
    
    print(f"Found {len(audio_files)} total audio files")
    
    # STEP 1: Split files FIRST (before any processing)
    # This ensures no data leakage - augmented copies won't leak into test set
    files_train_val, files_test, y_train_val, y_test = train_test_split(
        audio_files, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    
    val_ratio = val_size / (1 - test_size)
    files_train, files_val, y_train_files, y_val = train_test_split(
        files_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=random_state
    )
    
    print(f"\nStratified split (BEFORE augmentation):")
    print(f"  Train: {len(files_train)} files")
    print(f"  Val:   {len(files_val)} files")
    print(f"  Test:  {len(files_test)} files")
    
    augmenter = create_augmenter()
    
    # STEP 2: Process each split separately
    results = {}
    
    # Training data (WITH augmentation)
    print("\n" + "="*50)
    print("Processing TRAINING data (with augmentation)...")
    print("="*50)
    train_specs, train_labels, train_meta = [], [], []
    
    for filepath in tqdm(files_train, desc="Train"):
        try:
            # Original
            mel_spec, meta = process_single_file(filepath, augment=False)
            train_specs.append(mel_spec)
            train_labels.append(meta['emotion_idx'])
            train_meta.append({**meta, 'augmented': False})
            
            # Augmented copies (neutral gets extra due to class imbalance)
            n_aug = n_augmentations * 2 if meta['emotion'] == 'neutral' else n_augmentations
            for _ in range(n_aug):
                mel_spec_aug, _ = process_single_file(filepath, augment=True, augmenter=augmenter)
                train_specs.append(mel_spec_aug)
                train_labels.append(meta['emotion_idx'])
                train_meta.append({**meta, 'augmented': True})
        except Exception as e:
            print(f"Error: {e}")
    
    # Validation data (NO augmentation)
    print("\n" + "="*50)
    print("Processing VALIDATION data (NO augmentation)...")
    print("="*50)
    val_specs, val_labels, val_meta = [], [], []
    
    for filepath in tqdm(files_val, desc="Val"):
        try:
            mel_spec, meta = process_single_file(filepath, augment=False)
            val_specs.append(mel_spec)
            val_labels.append(meta['emotion_idx'])
            val_meta.append({**meta, 'augmented': False})
        except Exception as e:
            print(f"Error: {e}")
    
    # Test data (NO augmentation)
    print("\n" + "="*50)
    print("Processing TEST data (NO augmentation)...")
    print("="*50)
    test_specs, test_labels, test_meta = [], [], []
    
    for filepath in tqdm(files_test, desc="Test"):
        try:
            mel_spec, meta = process_single_file(filepath, augment=False)
            test_specs.append(mel_spec)
            test_labels.append(meta['emotion_idx'])
            test_meta.append({**meta, 'augmented': False})
        except Exception as e:
            print(f"Error: {e}")
    
    # Save each split
    for name, (specs, labels, meta) in [
        ('train', (train_specs, train_labels, train_meta)),
        ('val', (val_specs, val_labels, val_meta)),
        ('test', (test_specs, test_labels, test_meta))
    ]:
        X = np.array(specs)[..., np.newaxis]
        y = np.array(labels)
        
        print(f"\n{name.upper()}: X={X.shape}, y={y.shape}")
        
        np.save(os.path.join(output_dir, f'X_{name}.npy'), X)
        np.save(os.path.join(output_dir, f'y_{name}.npy'), y)
        with open(os.path.join(output_dir, f'metadata_{name}.pkl'), 'wb') as f:
            pickle.dump(meta, f)
    
    # Summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Train: {len(train_specs)} samples (with augmentation)")
    print(f"Val:   {len(val_specs)} samples (original only)")
    print(f"Test:  {len(test_specs)} samples (original only)")
    print(f"\nNote: Stratified random split - same speakers may appear in train/test")
    print(f"      For speaker-independent evaluation, use actor-based splits")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    data_dir = "data/raw"
    output_dir = "data/processed"
    
    print("Processing dataset with STRATIFIED RANDOM splits...")
    print("(Augmentation applied ONLY to training data - no data leakage)\n")
    
    process_dataset_stratified(data_dir, output_dir, n_augmentations=2)
    
    print("\nProcessing complete!")
