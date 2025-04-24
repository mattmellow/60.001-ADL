import os
from datetime import datetime

# Dataset paths
DATA_ROOT = "../data"  # Change this to your dataset directory
TRAIN_REAL_DIR = os.path.join(DATA_ROOT, "train/Real")
TRAIN_FAKE_DIR = os.path.join(DATA_ROOT, "train/Fake")
VAL_REAL_DIR = os.path.join(DATA_ROOT, "validation/Real")
VAL_FAKE_DIR = os.path.join(DATA_ROOT, "validation/Fake")
TEST_REAL_DIR = os.path.join(DATA_ROOT, "test/Real")
TEST_FAKE_DIR = os.path.join(DATA_ROOT, "test/Fake")

# Model parameters
INPUT_SIZE = 256  # Input image size
BACKBONE = "resnet34"  # Feature extractor backbone
EMBEDDING_DIM = 512  # Dimension of fingerprint embedding
DROPOUT_RATE = 0.4  # Dropout for regularization

# Training parameters
BATCH_SIZE = 16  # Adjust based on GPU memory
NUM_WORKERS = 10  # Number of data loading workers
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
WARMUP_EPOCHS = 2

# Mixed precision training
USE_AMP = True  # Set to True to use automatic mixed precision

# Checkpoints and logging
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
EXPERIMENT_LOGS = "experiment_logs"
EXPERIMENT_NAME = f"ganfingerprint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Hardware settings
DEVICE = "cuda"  # Use "cuda" for GPU, "cpu" for CPU

# Augmentation strength
AUG_STRENGTH = 0.5  # Strength of data augmentations (0-1)

# Reproducibility
SEED = 42  # Global seed for reproducibility
DETERMINISTIC = True  # Force deterministic operations
BENCHMARK = False # Set to false for reproducibility