import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = './data'
MODEL_PATH = os.getenv('MODEL_PATH', 'model/image_classifier.pth')
BATCH_SIZE = 32
EPOCHS = 5 # Increased for better accuracy

def train_model():
    # 1. Data Augmentation and Loading
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = train_dataset.classes

    # 2. Initialize Pre-trained Model (MobileNetV2)
    print("Loading pre-trained MobileNetV2...")
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    
    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 3. Training Loop with Validation
    print(f"Starting training on {device} for {EPOCHS} epochs...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation Phase
        model.eval()
        running_corrects = torch.tensor(0).to(device)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_acc = running_corrects.double() / len(val_dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Acc: {epoch_acc:.4f}")

        # 4. Save the best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            save_dict = {
                'state_dict': model.state_dict(),
                'class_names': class_names,
                'accuracy': float(best_acc)
            }
            torch.save(save_dict, MODEL_PATH)
            print(f"Improved model saved with accuracy: {best_acc:.4f}")

    print(f"Training complete. Best accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    train_model()
