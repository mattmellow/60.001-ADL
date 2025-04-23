
# Deepfake Detection with EfficientNet

This project uses **EfficientNet-B4** to detect deepfake images by identifying subtle visual manipulations that are often missed by traditional models.

EfficientNet’s main strength lies in its **compound scaling strategy**, which scales depth, width, and resolution together. This makes it especially effective at retaining **fine-grained spatial features**—a critical advantage when detecting nuanced inconsistencies in deepfakes.

We used a **pretrained EfficientNet-B4** and replaced the classifier head with a fully connected layer for **binary classification (real vs. fake)**. The model was trained using the **Adam optimizer** and **BCEWithLogitsLoss**, which is suited for binary outputs.

---

##  Training Strategy 

- **Input image size**: 260×260  
- **Two-phase gradual unfreezing**:
  - **Phase 1**: Trained only the classifier (`learning rate = 1e-3`)
  - **Phase 2**: Unfroze the classifier and **block 6** of the EfficientNet-B4 backbone (`learning rate = 5e-4`)
- This staged unfreezing approach allowed the model to adapt deeper layers without disrupting the pretrained feature representations, leading to improved performance without overfitting.

---

## Data Augmentation

To enhance generalization and reduce overfitting, we applied **data augmentation** using `torchvision.transforms` during training:

- **Resizing** to 260×260
- **Color jittering** (brightness, contrast, saturation adjustments)
- **Conversion to tensor**
- **Normalization** using ImageNet mean and standard deviation

For validation and testing, only resizing and normalization were applied to ensure evaluation consistency.

---

##  How to Run the Code

### 1. Install Required Libraries

```bash
pip install torch torchvision scikit-learn matplotlib pandas pillow
```


### 2. Train the Model

Open and run the notebook:

```
training_n_testing_notebooks/efficientnet_training.ipynb
```

It will automatically load the dataset and begin training using EfficientNet-B4 with the two-phase unfreezing strategy.

### 6. Test the Model

The trained weight file (e.g. `efficientnetb4_260_only2phases.pth`) is saved in:

```
models/efficientnetb4_260_only2phases.pth
```

It will be loaded when you run:

```
training_n_testing_notebooks/efficientnet_testing.ipynb
```

The notebook will load the model and output performance metrics like Accuracy, F1 Score, and AUC-ROC.

---

##  Required Libraries

Make sure the following libraries are installed:

```bash
pip install -r requirements.txt
```

---


##  Credits

Model architecture and dataset used from:
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [Kaggle Deepfake Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
