import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
import io
from src.config import MODEL_PATH

app = FastAPI(title="Image Classification API")

# Global variables for model and classes
model = None
class_names = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Image transformations for inference
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.on_event("startup")
def load_model():
    global model, class_names
    model_path = os.getenv('MODEL_PATH', 'model/image_classifier.pth')
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Please train the model first.")
        return

    # Load weights and class maps
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    
    # Rebuild model structure
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

@app.get("/health")
def health_check():
    """Endpoint for Docker health checks."""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an image file and returns the predicted class and confidence."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # 1. Error Handling: Check if file was provided
    if not file:
        return JSONResponse(status_code=400, content={"error": "No file uploaded."})

    # 2. Error Handling: Check MIME type
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Uploaded file is not a valid image format."})

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return JSONResponse(status_code=400, content={"error": "File is corrupted or not a valid image format."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Preprocess image and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence.item())
    }
