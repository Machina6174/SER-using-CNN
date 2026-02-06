# Speech Emotion Recognition using CNN

A deep learning project that classifies emotions from speech audio using a 2D Convolutional Neural Network trained on Mel-spectrograms.

---

## Author

| Name | Student ID |
|------|------------|
| **Daksh Gargi** | 2024B3A70888P |

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 89.93% |
| **Macro F1-Score** | 0.9023 |
| **Parameters** | 423,688 |
| **Emotions** | 8 classes |

### Per-Emotion Accuracy
| Emotion | Recall | Notes |
|---------|--------|-------|
| Neutral | 94% |  |
| Calm | 97% |   |
| Happy | 74% |  Confused with Surprised |
| Sad | 84% |  |
| Angry | 92% |  |
| Fearful | 95% |  |
| Disgust | 92% |  |
| Surprised | 95% |  |

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Machina6174/SER-using-CNN.git
cd SER-using-CNN
```

### 2. Create virtual environment

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
```bash
python scripts/download_dataset.py
```

### 5. Run inference
```bash
python predict.py data/raw/Actor_01/03-01-05-01-01-01-01.wav
```

**Expected output:**
```
Predicted emotion: ANGRY (97.5% confidence)
```

---

## Project Structure

```
SER-using-CNN/
├── data/
│   ├── raw/                    # RAVDESS audio files (24 actors)
│   └── processed/              # Preprocessed spectrograms
├── models/
│   └── best_model.keras        # Trained model weights
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis
├── scripts/
│   └── download_dataset.py    # Dataset downloader
├── src/
│   ├── preprocessing.py       # Audio processing pipeline
│   ├── model.py              # CNN architecture
│   └── train.py              # Training script
├── predict.py                 # Inference script
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── JOURNEY.md                # Development story
```

---

## Model Architecture

```
Input: (128, 130, 1) Mel-Spectrogram
    │
    ├── Conv2D(32) → BatchNorm → ReLU → MaxPool(2×2)
    ├── Conv2D(64) → BatchNorm → ReLU → MaxPool(2×2)
    ├── Conv2D(128) → BatchNorm → ReLU → MaxPool(2×2)
    ├── Conv2D(256) → BatchNorm → ReLU → MaxPool(2×2)
    │
    ├── Global Average Pooling
    ├── Dropout(0.6)
    ├── Dense(128, ReLU)
    ├── Dropout(0.4)
    │
    └── Dense(8, Softmax) → Output
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| **Dataset** | RAVDESS (2,880 files) |
| **Split** | 80% train / 10% val / 10% test |
| **Augmentation** | Noise, Pitch Shift, Time Stretch |
| **Augmentation Ratio** | 3.1x (training only) |
| **Optimizer** | Adam (lr=0.001) |
| **Loss** | Sparse Categorical Crossentropy |
| **Epochs** | ~30 (early stopping) |
| **Best Epoch** | 13 |
| **Batch Size** | 32 |
| **Class Weights** | Applied for imbalance |

### Data Augmentation
- **Gaussian Noise**: Random noise injection
- **Pitch Shifting**: ±4 semitones
- **Time Stretching**: 0.8x - 1.25x speed

---

## Usage Examples

### Basic Inference
```bash
python predict.py path/to/audio.wav
```

### With Verbose Output (all probabilities)
```bash
python predict.py path/to/audio.wav --verbose
```

### Custom Model Path
```bash
python predict.py audio.wav --model path/to/model.keras
```

---

## Known Limitations & Challenges

### Acoustic Ambiguity
| Confusion Pair | Why It Happens | Human Accuracy |
|----------------|----------------|----------------|
| Calm ↔ Sad ↔ Neutral | All low-arousal, quiet, slow speech | ~70% |
| Happy ↔ Surprised | Both high-energy, positive valence | ~80% |

> These confusions are **inherent to audio-only emotion recognition** — even trained human annotators struggle with these distinctions.

### Model Biases
- **Gender Gap**: 6.5% accuracy difference (93.2% female vs 86.6% male)
- **Happy on Males**: Only 61% recall (vs 85% female) — male "happy" is acoustically less distinct

### Dataset Constraints
- **Acted Speech**: Professional actors, not spontaneous emotions
- **Single Sentence**: All samples are "Kids are talking by the door"
- **English Only**: No multilingual support
- **Clean Audio**: No background noise, music, or overlapping speech

---

## Future Scope

| Enhancement | Expected Impact | Difficulty |
|-------------|-----------------|------------|
| **Attention Mechanisms** | Focus on emotional cues in speech | Medium |
| **Multi-Modal Fusion** | Combine audio + text + facial expressions | High |
| **Transfer Learning** | Use pretrained models (Wav2Vec2, HuBERT) | Medium |
| **Cross-Dataset Training** | Train on RAVDESS + CREMA-D + TESS | Medium |
| **Real-Time Streaming** | Process live audio input | Low |
| **Noise Robustness** | Handle real-world noisy environments | High |

---

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 24 professional actors (12 male, 12 female)
- 8 emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- ~2,880 audio files

---
