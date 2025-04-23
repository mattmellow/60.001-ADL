# Deepfake Detection with DenseNet-121 with FFT

This project uses **DenseNet-121 with FFT** to detect deepfake images by identifying subtle visual manipulations that are often missed by traditional models.

DenseNet's main strength lies in its **dense connectivity pattern**, where each layer is connected to every other layer. This allows the model to retain more fine-grained spatial features, which is critical when detecting nuanced inconsistencies in deepfakes.

We used a **pretrained DenseNet-121** and replaced the classifier head with a fully connected layer for **binary classification (real vs. fake)**. The input images undergo FFT to transform the image into frequencies. The model was trained using the **Adam optimizer** and **BCEWithLogitsLoss**, which is suited for binary outputs.

---

## Training Strategy

- **Input image size**: 256×256
- **Four-phase gradual unfreezing**:
  - **Phase 1**: Trained only the classifier (`learning rate = 1e-3`)
  - **Phase 2**: Unfroze the classifier and **block 4** of the DenseNet-121 backbone (`learning rate = 1e-4`)
  - **Phase 3**: Unfroze the classifier and **block 3 and 4** of the DenseNet-121 backbone (`learning rate = 1e-5`)
  - **Phase 4**: Unfroze all parameters of DenseNet-121 (`learning rate = 1e-6`)
- This staged unfreezing approach allowed the model to adapt deeper layers without disrupting the pretrained feature representations, leading to improved performance without overfitting.

---

## Data Augmentation

To enhance generalization and reduce overfitting, we applied **data augmentation** using `torchvision.transforms` during training:

- **Resizing** to 224×224
- **Color jittering** (brightness, contrast, saturation adjustments)
- **Conversion to tensor**
- **FFT Shift**
- **Normalization** using ImageNet mean and standard deviation

For validation and testing, only resizing and normalization were applied to ensure evaluation consistency.

---

## How to Run the Code

### 1. Install Required Libraries

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install pandas==2.2.3
```

### 2. Train the Model

Open and run the notebook:

```
training_n_testing_notebooks/densenet with FFT training.ipynb
```

It will automatically load the dataset and begin training using DenseNet-121 with the four-phase unfreezing strategy.

### 6. Test the Model

The trained weight file (e.g. `fftdensenet_epoch10.pth`) is saved in:

```
models/fftdensenet_epoch10.pth
```

It will be loaded when you run:

```
training_n_testing_notebooks/densenet with FFT training.ipynb
```

The notebook will load the model and output performance metrics like Accuracy, F1 Score, and AUC-ROC.

---

## Project Structure

Below is an overview of the project structure:

```
├── Densenet FFT
│   ├── models
│   │   └── fftdensenet_epoch10.pth
│   ├── training_n_testing_notebooks
│   │   ├── densenet with FFT testing.ipynb
│   │   └── densenet with FFT training.ipynb
│   ├── README.md
│   └── requirements.txt
```

- **DenseNet-121**: Contains the DenseNet-121 architecture, training, and testing notebooks, and the model weights for DenseNet-121 with FFT.
- **README.md**: Provides project details, instructions, and setup.
- **requirements.txt**: Lists all necessary libraries and dependencies for the project.

---

## Required Libraries

Make sure the following libraries are installed:

```bash
pip install -r requirements.txt
```

---

## Credits

Model architecture and dataset used from:

- [DenseNet](https://arxiv.org/abs/1608.06993)
- [Kaggle Deepfake Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
