# J-Dek — Custom Wake Word Detector

A lightweight, from-scratch neural wake word detection system built with PyTorch and deployed via ONNX Runtime for fast, dependency-light inference. Trained to recognize the custom wake phrase **"Jaydek"** from streaming microphone audio.

---

## Overview

This project implements an end-to-end wake word detection pipeline: audio preprocessing, a compact CNN classifier, ONNX export for portable inference, and a streaming detector service designed as a drop-in replacement for template-matching (DTW) based approaches.

**Pipeline at a glance:**

```
Raw audio (.wav)
   │
   ▼
Standardization (16kHz mono, 1.5s fixed window)
   │
   ▼
Log-Mel Spectrogram (128 mel bins, n_fft=1024, hop=512)
   │
   ▼
WakeWordCNN (3× Conv2D + BatchNorm + MaxPool → FC)
   │
   ▼
Sigmoid(logit) → threshold → wake word detected
```

---

## Features

- **Compact CNN architecture** — 3 convolutional blocks with batch normalization, ~64k parameters, fast enough for real-time CPU inference
- **ONNX export pipeline** — portable, framework-independent inference via `onnxruntime`, with optional BatchNorm folding via `onnxsim`
- **Streaming detector service** — sliding-window audio buffer with configurable inference interval, consecutive-hit confirmation, and cooldown to suppress false triggers
- **Energy gating** — cheap pre-inference silence detection to reduce unnecessary CPU load
- **Class-imbalance handling** — weighted sampling to fairly represent rare positive wake word clips against a much larger negative pool
- **Background noise augmentation** — synthetic noise mixing during training to improve real-world robustness
- **Checkpoint compatibility** — supports both legacy raw state_dict checkpoints and the current `{model, optimizer}` format, enabling seamless continued training

---

## Project Structure

```
.
├── model.py                    # WakeWordCNN architecture definition
├── spectrogram_generator.py    # WakeWordDataset: audio loading + feature extraction
├── data_standardize.py         # One-time preprocessing: resample, trim/pad to 1.5s
├── train.py                    # Training loop with weighted sampling + checkpointing
├── export_to_onnx.py           # PyTorch checkpoint → ONNX model conversion
└── services/
    └── onnx_wake_detector.py   # Streaming ONNX-based detector for production use
```

---

## Installation

```bash
git clone https://github.com/j-deku/wake_word_model.git
cd jaydek-wake-word
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install torch torchaudio soundfile librosa scikit-learn numpy tqdm
pip install onnx onnxsim onnxruntime
```

---

## Usage

### 1. Prepare your dataset

Organize raw `.wav` clips into three folders:

```
data/
├── positive/     # Clips of the wake word
├── negative/     # Clips of everything else (speech, silence, etc.)
└── background/   # Ambient noise used for augmentation
```

Standardize them to a consistent format (16kHz mono, fixed 1.5s length):

```bash
python data_standardize.py
```

### 2. Train the model

```bash
python train.py 
```

Training automatically resumes from `wake_word_model2.pth` if it exists, so re-running the script continues training rather than restarting from scratch.

### 3. Export to ONNX

```bash
python export_to_onnx.py --checkpoint wake_word_model2.pth --out wake_word_model.onnx
```

### 4. Run live detection

```python
from services.onnx_wake_detector import ONNXWakeWordDetector

detector = ONNXWakeWordDetector(
    model_path="wake_word_model.onnx",
    on_detect=lambda word, score: print(f"Detected '{word}' (confidence: {score:.2f})")
)

# Feed streaming audio frames (numpy arrays) as they arrive
detector.process(audio_frame)
```

---

## Model Architecture

| Layer | Output Shape | Details |
|---|---|---|
| Conv2D + BN + ReLU + MaxPool | 16 × 64 × 23 | 3×3 kernel, padding=1 |
| Conv2D + BN + ReLU + MaxPool | 32 × 32 × 11 | 3×3 kernel, padding=1 |
| Conv2D + BN + ReLU + MaxPool | 64 × 16 × 5  | 3×3 kernel, padding=1 |
| Flatten + Dropout(0.3) | 5120 | |
| Fully Connected | 128 | ReLU |
| Fully Connected | 1 | Raw logit output |

Input: `(batch, 1, 128, 47)` — a single-channel log-mel spectrogram (128 mel bins × 47 time frames, from a 1.5s @ 16kHz clip).

---

## Roadmap

- [x] Custom CNN baseline trained from scratch
- [x] ONNX export with BatchNorm folding
- [x] Streaming detector with hit-confirmation and cooldown
- [ ] Transfer-learning-based model (via [openWakeWord](https://github.com/dscripka/openWakeWord)) for improved generalization across speakers and environments
- [ ] Hard-negative mining loop from real-world false positives
- [ ] Threshold calibration via ROC analysis on held-out validation data

---

## Notes & Limitations

This model was trained on a relatively small, speaker-limited dataset. It performs well in controlled conditions but may be sensitive to background noise or unfamiliar voices — this is a known limitation of training a wake word classifier from scratch on limited data, and is actively being addressed via the transfer-learning approach listed in the roadmap above.

---

## License

[MIT](LICENSE) — feel free to adapt and build on this for your own custom wake word projects.
