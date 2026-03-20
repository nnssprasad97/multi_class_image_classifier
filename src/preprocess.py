import os
import shutil
import random
from torchvision.datasets import CIFAR10
from PIL import Image

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_RATIO

def setup_directories():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, split), exist_ok=True)

def preprocess_data():
    print("Downloading CIFAR-10 dataset...")
    # Download the dataset
    train_dataset = CIFAR10(root=RAW_DATA_DIR, train=True, download=True)
    test_dataset = CIFAR10(root=RAW_DATA_DIR, train=False, download=True)
    
    classes = train_dataset.classes
    print(f"Found {len(classes)} classes: {classes}")

    setup_directories()
    
    # Merge and resplit for demonstration of the requested logic
    all_data = []
    for img, label in train_dataset:
        all_data.append((img, label))
    for img, label in test_dataset:
        all_data.append((img, label))
        
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * TRAIN_RATIO)
    train_split = all_data[:split_idx]
    val_split = all_data[split_idx:]
    
    print(f"Splitting into {len(train_split)} train and {len(val_split)} val images...")
    
    for split_name, split_data in [('train', train_split), ('val', val_split)]:
        for i, (img, label_idx) in enumerate(split_data):
            class_name = classes[label_idx]
            class_dir = os.path.join(PROCESSED_DATA_DIR, split_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            img_path = os.path.join(class_dir, f"img_{i}.jpg")
            img.save(img_path)

    print(f"Dataset successfully preprocessed and saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    preprocess_data()
