# 🛡️ DeepShield: Multi-Signal Deepfake Forensic Suite
### *B.Tech CSE Final Year Project — PBEL*

DeepShield is a forensic analysis platform that uses a **Hybrid Multi-Modal Ensemble** of four neural networks and five heuristic signal processors to detect deepfakes with explainable confidence scores.

---

## 📂 Project Structure

```
deepfake_gan_detector/
├── detect_image.py          # Core detection engine (ensemble consensus + heuristics)
├── ui_app.py                # Tkinter GUI dashboard
├── classifier_model.py      # Multi-Branch ConvNeXt (RGB + FFT + Noise streams)
├── xade_model.py            # XADE EfficientNet-B4 model definition
├── gan_model.py             # DCGAN Generator + Discriminator
├── face_preprocessing.py    # RetinaFace alignment + augmentation pipeline
├── utils.py                 # Shared utility functions
├── train_classifier.py      # Training script — Multi-Branch ConvNeXt
├── train_gan.py             # Training script — DCGAN
├── train_pipeline.py        # End-to-end training pipeline
├── evaluate_model.py        # Evaluation + metrics generation
├── download_hf_model.py     # HuggingFace Swin Transformer downloader
├── download_xade_model.py   # XADE model downloader helper
├── requirements.txt         # Python dependencies
├── saved_models/            # ← Place downloaded model weights here (see below)
│   ├── deepfake_classifier.pth
│   ├── deepfake_model.pth
│   ├── hf_deepfake_model/
│   │   └── pytorch_model.bin
│   └── xade_model/
│       └── best_model.pt
├── classifier_curves.png    # Training accuracy/loss curves
├── confusion_matrix.png     # Model evaluation confusion matrix
└── roc_curve.png            # ROC-AUC curve
```

---

## ⚙️ Requirements

- **Python** 3.9+
- **CUDA-capable GPU** recommended (CPU inference supported but slow)

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📥 Pre-trained Model Weights

Model weights are **not bundled** in this repository (too large for GitHub). You must download them separately and place them inside `saved_models/`.

| File | Size | Download |
|---|---|---|
| `deepfake_classifier.pth` | ~112 MB | [Google Drive / Release](#) |
| `deepfake_model.pth` | ~113 MB | [Google Drive / Release](#) |
| `xade_model/best_model.pt` | ~183 MB | [Google Drive / Release](#) |
| `hf_deepfake_model/` | ~331 MB | Auto-downloaded (see below) |

> **Tip:** Replace the `#` links above with your actual Google Drive / HuggingFace links after uploading the `.pth` files.

### Auto-download the Swin Transformer (HuggingFace)
The HuggingFace Swin Transformer model is fetched automatically on first run. To pre-download it manually:
```bash
python download_hf_model.py
```
Or via Python:
```python
from transformers import AutoModelForImageClassification
AutoModelForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224')
```

### Directory layout after placing weights
```
saved_models/
├── deepfake_classifier.pth
├── deepfake_model.pth
├── hf_deepfake_model/
│   ├── config.json
│   └── pytorch_model.bin
└── xade_model/
    └── best_model.pt
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/deepfake_gan_detector.git
cd deepfake_gan_detector
pip install -r requirements.txt
```

### 2. Download Model Weights
Place the `.pth` / `.pt` files into `saved_models/` as shown above.

### 3. Launch the GUI Dashboard
```bash
python ui_app.py
```
Opens an interactive Tkinter window. Click **Browse** to load any face image and get a full forensic analysis with landmark mesh overlay.

### 4. Command-Line Inference
```bash
python detect_image.py --image path/to/image.jpg
```
Prints a JSON result with verdict, confidence score, and per-signal breakdown.

---

## 🧠 How It Works

### Detection Pipeline
1. **Face Detection & Alignment** — RetinaFace (ResNet-50) detects 5-point landmarks; Warp-Affine normalizes pose.
2. **Parallel Signal Extraction** — Image is sent simultaneously to 4 neural models and 5 heuristic processors.
3. **Ensemble Voting** — Weighted confidence consensus across all signals.
4. **EXIF Forensics** — Missing metadata boosts fake score; camera EXIF tags dampen false positives.
5. **Verdict** — `REAL FACE` / `DEEPFAKE` / `AI-GENERATED` with confidence %.

### Neural Models
| Model | File | Target |
|---|---|---|
| Multi-Branch ConvNeXt | `classifier_model.py` | RGB + FFT + Noise tri-stream fusion |
| XADE EfficientNet-B4 | `xade_model.py` | StyleGAN spatial artifacts |
| Swin Transformer | HuggingFace (auto) | Diffusion model inconsistencies |
| DCGAN Discriminator | `gan_model.py` | Generator fingerprint detection |

### Heuristic Signal Processors (`detect_image.py`)
| Signal | Target |
|---|---|
| Radial FFT | GAN checkerboard artifacts in frequency domain |
| ELA (Error Level Analysis) | Localized editing hotspots |
| Noise Residual | Sensor noise inconsistency |
| Patch Analysis | Face-blend boundary seams |
| Watermark Scan | AI generator attribution signatures |

---

## 📊 Performance

| Metric | Score |
|---|---|
| Accuracy | ~91.4% |
| Precision | ~89.7% |
| Recall | ~93.2% |
| AUC-ROC | ~0.967 |

*(Evaluated on held-out test split of the combined CIFAKE + Real-vs-Fake + FaceForensics++ dataset)*

---

## 🗃️ Training From Scratch (Optional)

Pre-trained weights are provided (see above). If you want to retrain:

1. **Train the classifier**:
   ```bash
   python train_classifier.py
   ```
2. **Train the GAN**:
   ```bash
   python train_gan.py
   ```
3. **Full pipeline**:
   ```bash
   python train_pipeline.py
   ```
4. **Evaluate**:
   ```bash
   python evaluate_model.py
   ```

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
facenet-pytorch>=2.5.3
tqdm>=4.65.0
retinaface-pytorch>=0.0.8
scipy>=1.10.0
pytorch-lightning>=2.1.0
tensorboard>=2.15.0
transformers>=4.31.0
huggingface_hub>=0.16.4
efficientnet_pytorch>=0.7.1
```

---

## 📜 License

This project is submitted as a B.Tech CSE Final Year Project (PBEL Division). All rights reserved by the authors.
