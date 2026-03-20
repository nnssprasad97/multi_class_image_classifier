# Multi-Class Image Classifier with Transfer Learning

This project implements an end-to-end image classification pipeline using transfer learning. It includes data preprocessing, model training with MobileNetV2, evaluation, and a containerized FastAPI for deployment.

## Project Structure
- `data/`: Processed training and validation data.
- `model/`: Saved model artifacts (.pth).
- `results/`: Evaluation metrics and plots.
- `src/`: Source code for the pipeline.
  - `preprocess.py`: Downloads and splits the CIFAR-10 dataset.
  - `train.py`: Fine-tunes MobileNetV2 with data augmentation.
  - `evaluate.py`: Calculates model performance metrics.
  - `api.py`: FastAPI server for inference.
- `Dockerfile` & `docker-compose.yml`: Containerization configuration.

## Setup Instructions

### Local Development
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```
4. Run the pipeline stages:
   ```bash
   python src/preprocess.py
   python src/train.py
   python src/evaluate.py
   ```

### Running with Docker
1. Ensure Docker and Docker Compose are installed.
2. Build and start the service:
   ```bash
   docker-compose up --build
   ```

## API Endpoints
- `GET /health`: Check service status.
- `POST /predict`: Upload an image file for classification.
  - Body: `file` (form-data)
  - Returns: `{"predicted_class": "...", "confidence": ...}`

## Project Decisions

### Dataset Selection
Initially, Caltech-101 was planned. However, due to upstream 404 errors during download, the project was successfully migrated to **CIFAR-10**. This ensures a reliable and repeatable preprocessing stage.

### Model Selection
**MobileNetV2** was selected for its efficiency and lightweight nature. It is particularly well-suited for containerized deployment with limited CPU/Memory resources. By leveraging transfer learning from a model pre-trained on ImageNet, we achieve significantly faster convergence and higher accuracy on the CIFAR-10 dataset than training from scratch.

### Hyperparameters and Training
- **Base Model**: MobileNetV2 (Pre-trained on ImageNet)
- **Optimizer**: Adam (Learning Rate: 0.001, 1e-4 for fine-tuning)
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 32
- **Epochs**: 10 (with gradual unfreezing and best-model checkpointing)
- **Augmentation**: Random Resized Crop (224), Random Horizontal Flip, Random Rotation (15°)

### Potential Improvements & Hyperparameter Tuning
To further improve accuracy, future iterations could explore:
- **Unfreezing More Layers**: Gradually unfreezing a larger portion of the MobileNetV2 base network.
- **Extended Training**: Increasing the number of epochs (e.g., 20-30) alongside an adaptive learning rate scheduler (like `CosineAnnealingLR`).
- **Advanced Augmentation**: Introducing `ColorJitter` or `RandomErasing` to improve model robustness to varied lighting and occlusions.

### Evaluation Summary
The project now includes a full validation loop during training, saving the model that achieves the **highest validation accuracy** rather than just the final epoch. The latest metrics in `results/metrics.json` reflect the performance of this optimized model.
