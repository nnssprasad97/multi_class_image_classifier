import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

def classify_image(image_filepath):
    if image_filepath is None:
        return "Please upload an image."
        
    try:
        with open(image_filepath, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, files=files)
            
        if response.status_code == 200:
            result = response.json()
            return f"**Predicted Class:** {result['predicted_class']}\n**Confidence:** {result['confidence']:.4f}"
        else:
            return f"Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"Failed to connect to API: {str(e)}"

# Create the Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath", label="Upload an Image"),
    outputs=gr.Markdown(label="Prediction Result"),
    title="Multi-Class Image Classifier",
    description="Upload an image to classify it using the fine-tuned MobileNetV2 model.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
