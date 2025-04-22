## CViT - Convolutional Vision Transformer for Deepfake Detection

This repository implements a CViT-based image classification pipeline to detect deepfake images using a hybrid architecture combining convolutional and transformer layers.

---

### 🧠 Model Overview

CViT (Convolutional Vision Transformer) is a Transformer-based model that learns patch-wise attention, enhanced by convolutional embeddings. It is optimized for binary classification: **REAL** vs **FAKE** images.

---

### Link To Download Best Model For Weight
https://sutdapac-my.sharepoint.com/:u:/g/personal/edward_tang_mymail_sutd_edu_sg/EaGLH-ZShP9Il36Ak0SA0R0BBaZZGZed80enb4ANDkMe0A?e=8bw1Fq

---

### 📁 Project Structure

```
.
├── cvit_train.py              # Training script
├── cvit_infer_images.py       # Inference script
├── data                       # This is where you put your dataset
├── helpers
│   └──augmentation.py         # Strong image augmentation setup
│   └──loader.py               # Torch DataLoader setup with albumentations
├── model/
│   └── cvit.py                # CViT architecture (default version) [ReLU and BatchNorm removed]
│   └── cvit-relubatch.py      # CViT with ReLU and BatchNorm included
└── weight/                    # Folder for saving trained model weights
│   └── cvit_best_model.pth    # Best performing weight. Use with default cvit.py [ReLU and BatchNorm removed]
```

---
### 🔄 Using Different CViT Versions

By default, `cvit.py` is used for the architecture without ReLU and BatchNorm.  
If you would like to use the version **with** ReLU and BatchNorm:
1. Rename `cvit-relubatch.py` to `cvit.py`
2. Rename or move the original `cvit.py` (the one without ReLU and BatchNorm) to a backup name (e.g., `cvit-norelu.py`)

Vice versa, if you want to switch back:
1. Rename `cvit-norelu.py` back to `cvit.py`
2. Save or rename `cvit-relubatch.py` as needed

Ensure that only **one** `cvit.py` exists in the `model/` directory at a time to avoid import conflicts.


---

### ⚙️ Requirements

- Python 3.7+
- PyTorch
- torchvision
- albumentations
- scikit-learn
- matplotlib
- seaborn
- Pillow

```bash
pip install -r requirements.txt
```

---

### 🚀 Training

```bash
python cvit_train.py -e 30 -d ./data -b 32 -l 0.00005 -w 0.0000001 
```

**Arguments**:
- `-e` or `--epoch`: Number of epochs
- `-d` or `--dir`: Directory with `train/`, `validation/`, and `test/`
- `-b` or `--batch`: Batch size
- `-l` or `--rate`: Learning rate
- `-w` or `--wdecay`: Weight decay
- `-t` or `--test`: Run test after training

---

### 🔍 Inference

```bash
python cvit_infer_images.py --folder ./data/test --weights ./weight/cvit_deepfake_detection_xxx.pth
```

- Saves predictions to `image_predictions.json`
- Outputs confusion matrix image `confusion_matrix.png`

---

### 🔬 Data Augmentation

Defined in `augmentation.py` using Albumentations:
- Horizontal/Vertical flips
- Random rotation and transpose
- CLAHE, brightness/contrast shift
- Gaussian noise
- Hue/Saturation/Value adjustment

---

### 🧱 Architecture Summary

**CViT Configuration**:
- **Input Image Size**: 224x224
- **Patch Size**: 4x4
- **Channels**: 512
- **Embedding Dimension**: 1024
- **Transformer Depth**: 6 layers
- **Attention Heads**: 8
- **MLP Hidden Dim**: 2048
- **Output**: 2 classes (REAL or FAKE)

The architecture uses:
- Patch embedding with convolutions
- Multi-head self-attention
- Residual connections
- Pre-normalization using LayerNorm
- MLP block for final classification

---

### 📊 Outputs

- Training loss/accuracy per epoch
- Validation and test evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- `.pth` files for model weights

---

### 📄 Notes

- Uses GPU if available (`cuda` device)
- Ideal for deepfake detection use-cases
- Extendable for other binary image classification problems
- The **best-performing weights** in this repository were obtained using the version **without** ReLU and BatchNorm (i.e., the default `cvit.py`).

---