"""
Data loading utilities for the deepfake detector with reproducibility features.
"""
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.augmentations import JPEGCompression, AddNoiseGaussian, VariableBlur
# from torchvision.transforms import functional as F
import config


def worker_init_fn(worker_id):
    """
    Initialize worker with a fixed seed based on epoch and worker id.
    This ensures reproducible data loading across different runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DeepfakeDataset(Dataset):
    """Dataset for loading real and fake images with reproducible behavior."""
    
    def __init__(self, real_dir, fake_dir, transform=None, phase='train', seed=42):
        """
        Args:
            real_dir (str): Directory with real images
            fake_dir (str): Directory with fake images
            transform: PyTorch transform to apply to images
            phase (str): 'train', 'val', or 'test'
            seed (int): Random seed for reproducibility
        """
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        self.phase = phase
        self.seed = seed
        
        # Set seed for reproducibility
        random_state = random.getstate()
        random.seed(self.seed)
        
        # Get all image paths and labels
        self.real_imgs = [os.path.join(real_dir, f) for f in sorted(os.listdir(real_dir)) if self._valid_file(f)]
        self.fake_imgs = [os.path.join(fake_dir, f) for f in sorted(os.listdir(fake_dir)) if self._valid_file(f)]
        
        # Ensure balanced classes
        if phase == 'train':
            min_class_size = min(len(self.real_imgs), len(self.fake_imgs))
            self.real_imgs = random.sample(self.real_imgs, min_class_size)
            self.fake_imgs = random.sample(self.fake_imgs, min_class_size)
        
        self.image_paths = self.real_imgs + self.fake_imgs
        self.labels = [1] * len(self.real_imgs) + [0] * len(self.fake_imgs)  # 1 for real, 0 for fake
        
        # Shuffle data in a reproducible way
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)
        
        # Restore random state
        random.setstate(random_state)
    
    def _valid_file(self, filename):
        """Check if the file is an image file."""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            # For test/val, apply deterministic transforms
            if self.phase != 'train':
                # For reproducible validation/testing
                torch_seed_state = torch.get_rng_state()
                torch.manual_seed(self.seed + idx)  # Deterministic per sample
                image = self.transform(image)
                torch.set_rng_state(torch_seed_state)
            else:
                # For training, apply transforms normally
                image = self.transform(image)
        
        return image, label

# New get_transform(phase) function for custom augmentations.py
def get_transforms(phase):
    """
    Get the transformations for each dataset phase.
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]   # ImageNet std
    
    if phase == 'train':
        # PIL Image transforms (before ToTensor)
        pil_transforms = [
            transforms.RandomResizedCrop(config.INPUT_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomRotation(10)
            ], p=0.3),
            transforms.ColorJitter(
                brightness=0.2 * config.AUG_STRENGTH,
                contrast=0.2 * config.AUG_STRENGTH,
                saturation=0.2 * config.AUG_STRENGTH,
                hue=0.1 * config.AUG_STRENGTH
            ),
            # Custom PIL Image augmentations
            transforms.RandomApply([JPEGCompression(quality_range=(60, 95))], p=0.4),
            transforms.RandomApply([AddNoiseGaussian(std_range=(0.01, 0.05))], p=0.3),
            transforms.RandomApply([VariableBlur(radius_range=(0.1, 1.5))], p=0.3),
            # Convert to tensor
            transforms.ToTensor(),
        ]
        
        # Tensor transforms (after ToTensor)
        tensor_transforms = [
            # Random erasing (operates on tensors)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            # Normalize
            transforms.Normalize(mean=mean, std=std)
        ]
        
        # Combine all transforms
        return transforms.Compose(pil_transforms + tensor_transforms)
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(config.INPUT_SIZE),
            transforms.CenterCrop(config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def get_dataloaders(seed=42):
    """
    Create reproducible data loaders for train, validation, and test sets.
    
    Args:
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Set up transforms
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    test_transform = get_transforms('test')
    
    # Create datasets with fixed seed
    train_dataset = DeepfakeDataset(
        config.TRAIN_REAL_DIR, 
        config.TRAIN_FAKE_DIR, 
        transform=train_transform, 
        phase='train',
        seed=seed
    )
    
    val_dataset = DeepfakeDataset(
        config.VAL_REAL_DIR, 
        config.VAL_FAKE_DIR, 
        transform=val_transform, 
        phase='val',
        seed=seed
    )
    
    test_dataset = DeepfakeDataset(
        config.TEST_REAL_DIR, 
        config.TEST_FAKE_DIR, 
        transform=test_transform, 
        phase='test',
        seed=seed
    )
    
    # Create reproducible generators
    g_train = torch.Generator()
    g_train.manual_seed(seed)
    
    g_val = torch.Generator()
    g_val.manual_seed(seed)
    
    g_test = torch.Generator()
    g_test.manual_seed(seed)
    
    # Create samplers instead of using shuffle=True
    train_sampler = torch.utils.data.RandomSampler(
        train_dataset, 
        replacement=False, 
        generator=g_train
    )
    
    # Create data loaders with fixed workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,  # Use sampler instead of shuffle=True
        num_workers=config.NUM_WORKERS,
        worker_init_fn=worker_init_fn,  # Initialize workers with deterministic seed
        pin_memory=True,
        drop_last=True,
        generator=g_train  # For any remaining randomness in DataLoader
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation data
        num_workers=config.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        generator=g_val
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # No need to shuffle test data
        num_workers=config.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        generator=g_test
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_stats():
    """
    Print dataset statistics.
    """
    print("Dataset Statistics:")
    print(f"Train: {len(os.listdir(config.TRAIN_REAL_DIR))} real, {len(os.listdir(config.TRAIN_FAKE_DIR))} fake")
    print(f"Validation: {len(os.listdir(config.VAL_REAL_DIR))} real, {len(os.listdir(config.VAL_FAKE_DIR))} fake")
    print(f"Test: {len(os.listdir(config.TEST_REAL_DIR))} real, {len(os.listdir(config.TEST_FAKE_DIR))} fake")