import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from srgan import SRGAN

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset and data loader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root="data/train", transform=transform)
data_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
srgan = SRGAN(lr=LR)
srgan.generator.to(DEVICE)
srgan.discriminator.to(DEVICE)

def train():
    """Trains the SRGAN model."""
    for epoch in range(EPOCHS):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        for i, (high_res, _) in enumerate(data_loader):
            low_res = torch.nn.functional.interpolate(high_res, scale_factor=0.25, mode='bicubic', align_corners=False)
            high_res, low_res = high_res.to(DEVICE), low_res.to(DEVICE)
            
            g_loss, d_loss = srgan.train_step(high_res, low_res)
            g_loss_epoch += g_loss
            d_loss_epoch += d_loss
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(data_loader)}], G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
        
        print(f"Epoch {epoch+1} completed. Avg G Loss: {g_loss_epoch / len(data_loader):.4f}, Avg D Loss: {d_loss_epoch / len(data_loader):.4f}")
        
        # Save model checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(srgan.generator.state_dict(), f"checkpoints/generator_epoch_{epoch+1}.pth")
            torch.save(srgan.discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}.")

if __name__ == "__main__":
    train()