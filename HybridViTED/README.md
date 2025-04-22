
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
├── main_train.py              # Main training and testing script
├── config.py                  # Configuration loader
├── config.yaml                # Training and model config file
│
├── dataset/
│   └── loader.py              # Data loading and splitting utilities
│
├── model/
│   ├── genconvit_ed.py        # ViTED model architecture (encoder-decoder + ViT)
│   └── model_embedder.py      # SwinTransformer wrapper for patch embedding
│
├── train/
│   └── train_ed.py            # Training and validation loop
│
├── weight/                    # Saved model weights (.pth) and training logs (.pkl)
├── images/                    # Input images for inference
└── *.png / *.jpg              # Example data for testing
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

### From Scratch

```bash
python HybridViTED_train.py --d kaggle_train_data --e 25 -t y
```

**Arguments**:
- `--d`: dataset directory
- `--e`: number of epochs
- `--p`: path to pretrained model
- `--t`: test after training

---

## 📈 Evaluation Metrics

The model evaluates performance using:

- Accuracy
- Precision
- Recall
- F1 Score

These are logged in the console and saved in `.pkl` logs.

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
python HybridViTED_predict.py --image path/to/image.png --model weight/genconvit_ed.pth
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

- Based on concepts from [GANFingerprints](https://github.com) and [Swin Transformer](https://github.com/microsoft/Swin-Transformer).
- Trained on Kaggle's deepfake image datasets.
- Inspired by Vision Transformer and hybrid CNN-ViT models.
