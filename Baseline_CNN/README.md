# Deepfake Detection with Baseline CNN

This project uses a **Baseline CNN** to detect deepfake images by identifying subtle visual manipulations. The model is based on a simple Convolutional Neural Network architecture with two convolutional layers, followed by two fully connected layers.

The **baseline CNN** architecture was chosen for its simplicity. It serves as a foundation model against which more complex architectures can be compared. The goal was to establish the minimum performance baseline before adding more advanced features like transfer learning and Fourier transformations.

---

## Training Strategy

- **Input image size**: 224×224
- **Architecture**:

  - **First convolutional layer**: 3×3 kernel, 32 output channels
  - **Second convolutional layer**: 3×3 kernel, 64 output channels
  - **Max pooling**: Applied after each convolutional layer
  - **Fully connected layers**: 128 hidden units, followed by a final output layer for binary classification (real vs. fake)
- **Training Setup**:

  - **Optimizer**: Adam optimizer
  - **Loss function**: Binary Cross-Entropy with logits (BCEWithLogitsLoss) for binary classification
  - **Learning rate**: 1e-3
  - **Epochs**: 15 (Adjustable depending on the required performance)

This model trains with simple image augmentation, including resizing, color jittering, and normalization, designed to help the model generalize better to different image conditions.

---

## Data Augmentation

To enhance generalization and reduce overfitting, we applied **data augmentation** during training:

- **Resizing** to 224×224
- **Color jittering** (brightness, contrast, saturation adjustments)
- **Conversion to tensor**
- **Normalization** using ImageNet mean and standard deviation

For validation and testing, only resizing and normalization were applied to ensure consistent evaluation.

---

## How to Run the Code

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Required Libraries

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install pandas==2.2.3
```

### 3. Train the Model

Open and run the notebook:

```
training_n_testing_notebooks/base CNN training.ipynb
```

It will automatically load the dataset and begin training the Baseline CNN model.

### 4. Test the Model

After training, the model weight file (e.g. `baseCNN_epoch_3.pth`) is saved in:

```
models/baseCNN_epoch_3.pth
```

It can be loaded using:

```
training_n_testing_notebooks/base CNN testing.ipynb
```

The notebook will output performance metrics like Accuracy, Precision, Recall, and F1 Score.

---

## Project Structure

Below is an overview of the project structure:

```
├── Baseline CNN
│   ├── models
│   │   └── baseCNN_epoch_3.pth
│   ├── training_n_testing_notebooks
│   │   ├── base CNN testing.ipynb
│   │   └── base CNN training.ipynb
│   ├── README.md
│   └── requirements.txt
```

- **Baseline CNN**: Contains the baseline CNN model architecture, training, and testing notebooks, and the model weights.
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

- [Deepfake Detection Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

---

This **README.md** provides an overview and instructions for using the **Baseline CNN** model for deepfake detection. It explains how to set up the environment, train the model, and test it, along with details about the architecture and data preprocessing steps used in this project.
