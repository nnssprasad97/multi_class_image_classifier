import os
from dotenv import load_dotenv

load_dotenv()

# Directories
DATA_DIR = os.getenv('DATA_DIR', './data')
PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR', './data')
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR', './data_raw')
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')

# Model
MODEL_PATH = os.getenv('MODEL_PATH', 'model/image_classifier.pth')

# Training Hyperparameters
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
EPOCHS = int(os.getenv('EPOCHS', '10')) # Increased for better accuracy with gradual unfreezing
TRAIN_RATIO = float(os.getenv('TRAIN_RATIO', '0.8'))

# API
API_PORT = int(os.getenv('API_PORT', '8000'))
