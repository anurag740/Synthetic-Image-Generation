import os
import torch
from PIL import Image
from torchvision import transforms

# Define paths
input_dir = "images/"
output_dir = "outputs/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts image to tensor and normalizes it
])

def preprocess_image(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)

    # Convert tensor back to an image and save
    image_preprocessed = transforms.ToPILImage()(tensor)
    image_preprocessed.save(output_path)

# Process each image in the input folder
for i in range(1, 4):  # Assuming 3 images: synthetic_image_1, 2, 3
    input_path = f"{input_dir}synthetic_image_{i}.png"
    output_path = f"{output_dir}preprocessed_image_{i}.png"
    
    preprocess_image(input_path, output_path)
    print(f"Saved: {output_path}")
