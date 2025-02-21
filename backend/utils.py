import torch
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_path, device):
    """Loads an image and converts it to a tensor for model processing."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension

def save_image(tensor, output_path):
    """Converts a tensor back to an image and saves it to a file."""
    transform = transforms.ToPILImage()
    image = transform(tensor.squeeze(0).cpu())  # Remove batch dimension
    image.save(output_path)

def upscale_image(generator, image_tensor, device):
    """Uses the trained generator model to upscale an image."""
    generator.to(device)
    generator.eval()
    with torch.no_grad():
        sr_tensor = generator(image_tensor)
    return sr_tensor