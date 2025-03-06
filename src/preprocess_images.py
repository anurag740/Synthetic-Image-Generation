import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to tensor and normalizes [0, 1]
    ])
    tensor = transform(image)
    return tensor

for i in range(3):
    tensor = preprocess_image(f"synthetic_image_{i+1}.png")
    print(f"Image {i+1} Tensor Shape: {tensor.shape}")
