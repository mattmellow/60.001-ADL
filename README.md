# 🧠 ADL Deepfakers: Deepfake Detection Using CNNs, Transformers, and GANFingerprints

## 📌 Introduction

Deepfakes are AI-generated synthetic media where a person’s likeness is manipulated to portray actions or speech they never performed. These forgeries, driven by GANs and transformer models, pose a real threat—especially on social media—by spreading misinformation, compromising reputations, and eroding public trust. High-profile incidents involving deepfake impersonations of political figures highlight the urgency for reliable detection methods.

## 🌟 Objectives

We aim to develop a robust image-based deepfake classification model capable of:

- Detecting deepfakes created by state-of-the-art generation models.
- Adapting to newer, more subtle manipulation techniques.
- Minimizing **false negatives** (i.e., missed deepfakes) to curb their spread.

## 🧪 Methodology

Our approach is structured around:

- **GANFingerprint**: Our best-performing model, detecting unique GAN artifacts.
- **Baseline CNN**: Simple convolutional model for benchmarking.
- **Transfer Learning**: Models like ResNet18, EfficientNet, DenseNet.
- **Hybrid Architectures**: Including CViT and ViTED combining CNNs and Transformers.
- **FFT Preprocessing**: Frequency domain analysis for subtle anomaly detection.
- Extensive **training iteration analysis** and **hyperparameter tuning** with recall-focused evaluation metrics.

## 🤖 Models Explored

- `GANFingerprint`: Customized ResNet-based model leveraging spatial + frequency analysis.
- `Baseline CNN`: Two-layer convolutional classifier.
- `ResNet18`: Lightweight residual model with skip connections.
- `EfficientNet-B4`: Scaled CNN optimized for resolution, depth, and width.
- `DenseNet121`: Feature-dense model with layer-wise connectivity.
- `CViT`: Combines convolutional stems with transformer encoders.
- `ViTED`: Encoder-decoder pipeline using CNN + ViT with dual inputs.

## 🗂️ Project Structure

```
.
├── data/
│   └── Place your dataset here (see below)
├── Baseline CNN/
├── CViT/
├── Densenet FFT/
├── Efficientnet/
├── HybridViTED/
├── Deepfake-detection-GANFingerprint
└── README.md
```

## 🗃️ Dataset and weight files Instructions

We used the **[Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)** dataset (based on OpenForensics).

All weight files, as well as the zipped dataset are made available (only to SUTD email holders) via this Sharepoint folder: https://sutdapac.sharepoint.com/:f:/s/SDSMushroomersADLDeepfakers/EjiZHgFizgxPhs5HpNmRlwgButzHrzzIq0NWhi1Q7nymWw?e=UPkdXs

1. Download and unzip the dataset.
2. Place it inside the `data/` directory.
3. The dataset should already have the following structure:

```plaintext
data/
├── train/
├── validation/
└── test/
```

Each subfolder should include labeled `Real` and `Fake` directories with image files.

## 📁 Running the Code

Every model folder contains its own `README.md` with:

- Environment setup instructions
- Training and evaluation scripts
- Expected input structure
- Best-performing checkpoints

Refer to each folder depending on the architecture you'd like to explore.

## 💡 Conclusion

The **GANFingerprint** model emerged as the top performer in terms of accuracy, F1-score, and recall. Our investigation highlights how combining architectural experimentation, frequency-based preprocessing, and transfer learning can significantly improve deepfake detection. However, computational resources and data diversity remain major constraints.

This repository serves not only as a deepfake detector but as a testbed for experimenting with hybrid architectures in computer vision and adversarial learning.

---

📬 For inquiries or contributions, feel free to reach out via Issues or Pull Requests.

