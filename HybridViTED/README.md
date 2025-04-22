
# 🔍 ViTED: Vision Transformer Encoder-Decoder for Deepfake Detection

This project explores a **deepfake detection pipeline** using a **hybrid Encoder-Decoder and Vision Transformer (ViT)** architecture. The core idea is to reconstruct images through an encoder-decoder network and use both original and reconstructed images to make binary classifications (real/fake) via a transformer-based model.

---

## 🧠 Model Overview

**ViTED** integrates three main components:

1. **Encoder**  
   A CNN-based feature extractor (initially custom, later upgraded to ResNet18).

2. **Decoder**  
   Upsamples feature maps back to the image space using transposed convolutions.

3. **Backbone Classifier (ViT)**  
   A Vision Transformer receives both real and reconstructed images to learn semantic differences. It uses Swin Transformer as the patch embedding module, combined via a `HybridEmbed` wrapper.

---

## 🏗️ Project Structure

```
project_root/
│
├── HybridViTED_train.py          # Main training and evaluation script (replaces main_train.py)
├── HybridViTED_predict.py        # Predict single image using trained ViTED model
├── HybridViTED_get_metrics.py    # Extract metrics from saved .pkl files
│
├── dataset/
│   └── loader.py                 # Data loader and pre-processing pipeline
│
├── model/
│   ├── genconvit_ed.py           # Hybrid Encoder-Decoder ViT architecture
│   ├── model_embedder.py         # Swin Transformer wrapper for embedding
│   ├── config.py                 # Configuration loader
│   └── config.yaml               # YAML config for training and model settings
│
├── train/
│   └── train_ed.py               # Training and validation functions
│
├── weight/                       # Saved model weights (.pth) and training metrics (.pkl)
│
├── images/                       # Folder for storing test images used during prediction
│   └── *.png / *.jpg             # Example images for testing inference
│
└── README.md                     # Project instructions, usage, architecture, and logs

```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
model:
  backbone: resnet18
  embedder: swin_tiny_patch4_window7_224
  latent_dims: 12544

learning_rate: 0.001
weight_decay: 0.001
min_val_loss: 10
batch_size: 32
img_size: 224
```

---

## 🚀 How to Train

### From Scratch for the best model

```bash
python HybridViTED_train.py --d kaggle_train_data --e 20 -t y
```

**Arguments**:
- `--d`: dataset directory
- `--e`: number of epochs

---

## 📈 Evaluation Metrics

The model evaluates performance using:

- Accuracy
- Precision
- Recall
- F1 Score

These are logged in the console and saved in `.pkl` logs.

```bash
python HybridViTED_get_metrics.py --model weight/genconvit_ed.pth --data path/to/image --batch_size 32
```

---

## 🔄 Iterations & Tuning

| Iteration | Change Summary                       | Accuracy | Precision | Recall | F1 Score |
|-----------|--------------------------------------|----------|-----------|--------|----------|
| Train 1   | Custom CNN baseline                  | 0.8123   | 0.8717    | 0.7292 | 0.7941   |
| Train 2   | Longer epochs → Overfitting          | 0.5388   | 0.9948    | 0.0713 | 0.1331   |
| Train 3   | Switched to ResNet18                 | 0.7782   | 0.7357    | 0.8631 | 0.7944   |
| Train 4   | Tuned LR (ResNet + 0.001 LR)         | 0.9072   | 0.9145    | 0.8969 | 0.9056   |
| Train 5   | BatchNorm in Decoder                 | 0.8029   | 0.7965    | 0.8099 | 0.8032   |
| Train 6   | Deep FC + Dropout (0.5)              | 0.5036   | 0.0000    | 0.0000 | 0.0000   |
| Train 7   | Simplified FC + Dropout (0.3)        | 0.4964   | 0.4964    | 1.0000 | 0.6634   |

---

## 🧪 Inference Example

```bash
python HybridViTED_predict.py --image path/to/image --model weight/genconvit_ed.pth
```

Optional: add `--label_encoder your_encoder.pkl` for class names.

---

## 🧱 Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main packages:
- PyTorch
- `timm` (for ViTs)
- OpenCV / PIL
- scikit-learn
- Albumentations

---

## 📬 Acknowledgements

This project is adapted from the original [GenConViT GitHub repository](https://github.com/erprogs/GenConViT), which implements the Generative Convolutional Vision Transformer for deepfake detection. Significant modifications were made, including the integration of resnet18 into hybrid encoder-decoder architecture, iterative training experiments, and extended evaluation tooling for comprehensive performance analysis.
- Trained using the [Deepfake and Real Images Dataset from Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images).
- Inspired by Vision Transformer and hybrid CNN-ViT models.
