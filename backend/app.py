from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import torch
from generator import Generator
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import io

app = FastAPI()

# Load pre-trained generator model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator()
generator.load_state_dict(torch.load("checkpoints/generator_epoch_100.pth", map_location=DEVICE))
generator.to(DEVICE)
generator.eval()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Handles image upload, applies super-resolution, and returns the result."""
    
    # Read image bytes and convert to PIL Image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Preprocess: Convert image to tensor and move to device
    transform = ToTensor()
    lr_image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    # Generate high-resolution image
    with torch.no_grad():
        sr_image = generator(lr_image)
    
    # Convert tensor back to PIL Image
    sr_image = sr_image.squeeze(0).cpu()
    transform_to_pil = ToPILImage()
    sr_pil = transform_to_pil(sr_image)
    
    # Save output image
    output_path = "output/super_resolved.png"
    sr_pil.save(output_path)
    
    return FileResponse(output_path, media_type="image/png")

@app.get("/")
def home():
    return {"message": "Welcome to the Super-Resolution API! Upload an image at /upload/"}