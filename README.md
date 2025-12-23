# Gesture Recognition Models

This project builds two deep-learning models for real‑time gesture recognition using video sequences:

1. **2D CNN + RNN (MobileNet + GRU)**
2. **3D CNN**

Both models classify **5 hand gestures** from video clips of 30 frames each.

---

## ✨ Features
- Custom **video frame generator**
- **Transfer learning** with MobileNet (frozen base)
- **GRU-based** sequence modeling
- **3D CNN** for spatiotemporal features
- **Model checkpoints** & **ReduceLROnPlateau**
- Accuracy/Loss plots for training vs. validation

---

## 📁 Dataset
- Input: folders of **30‑frame** sequences
- Image size: **128 × 128 × 3**
- Labels: `train.csv` and `val.csv`

Folder layout:
```
Project_data/
├── train/
│   └── <video_folder_1>/frame_000.jpg ... frame_029.jpg
└── val/
    └── <video_folder_k>/frame_000.jpg ... frame_029.jpg
```

---

## 🧠 Models

### 1) 2D CNN + RNN
- **MobileNet** (imagenet, `include_top=False`) via `TimeDistributed`
- GRU(64) → Dropout(0.2) → Dense(64, relu) → Dense(5, softmax)
- Lightweight & webcam‑friendly

### 2) 3D CNN
- Conv3D(32) → MaxPool3D → Conv3D(64) → MaxPool3D → Dropout(0.2)
- Flatten → Dense(128, relu) → Dense(5, softmax)

---

## 🚀 Training
- Epochs: **25**
- Batch size: **32**
- Callbacks:
  - `ModelCheckpoint` (best `val_categorical_accuracy`)
  - `ReduceLROnPlateau` (monitor `val_loss`)
- Steps per epoch computed from sequence counts

---

## 📊 Evaluation
- Accuracy & loss curves for both models using Matplotlib
- Console logs show per‑epoch train/val metrics

---

## 🛠 Requirements
- Python ≥ 3.8
- TensorFlow / Keras
- NumPy
- OpenCV‑Python
- Matplotlib

Install:
```bash
pip install tensorflow numpy opencv-python matplotlib
```

---

## ▶️ Usage
1. Place dataset under `Project_data/train` and `Project_data/val` with 30 frames per folder.
2. Run the script to:
   - Load data via the custom generator
   - Train **2D CNN + RNN** and **3D CNN**
   - Save best checkpoints in timestamped folders
3. Use the plotting functions to visualize accuracy/loss.

---

## ⚠️ Notes
- Ensure each folder has **exactly 30 frames**.
- Images are **resized to 128×128** and **normalized to [0,1]** inside the generator.
- MobileNet base is **frozen** for faster training and to prevent overfitting; you can unfreeze for fine‑tuning if needed.

---

## 📄 License
This project is for educational and research purposes.
