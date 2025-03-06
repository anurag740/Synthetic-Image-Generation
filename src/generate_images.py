from diffusers import StableDiffusionPipeline
import torch

# Load pre-trained Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "a serene sunset over a futuristic city"
for i in range(3):
    image = pipe(prompt).images[0]
    image.save(f"synthetic_image_{i+1}.png")
    print(f"Image {i+1} saved.")
