
import torch
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator

class SRGAN:
    """
    SRGAN (Super-Resolution GAN) framework that combines the Generator and Discriminator,
    and facilitates training using Adversarial Loss and Content Loss.
    """
    def __init__(self, lr=1e-4):
        super(SRGAN, self).__init__()
        
        # Initialize Generator and Discriminator
        self.generator = Generator()
        self.discriminator = Discriminator()
        
        # Define Loss Functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for Discriminator
        self.content_loss = nn.MSELoss()  # Mean Squared Error Loss for content similarity
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.9, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
    
    def train_step(self, real_images, low_res_images):
        """
        Performs one training step for both Generator and Discriminator.
        """
        batch_size = real_images.size(0)
        valid_labels = torch.ones((batch_size, 1), requires_grad=False)
        fake_labels = torch.zeros((batch_size, 1), requires_grad=False)
        
        # Train Generator
        self.optimizer_G.zero_grad()
        fake_images = self.generator(low_res_images)
        adversarial_loss = self.adversarial_loss(self.discriminator(fake_images), valid_labels)
        content_loss = self.content_loss(fake_images, real_images)
        g_loss = content_loss + 1e-3 * adversarial_loss  # Weighted sum of losses
        g_loss.backward()
        self.optimizer_G.step()
        
        # Train Discriminator
        self.optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(real_images), valid_labels)
        fake_loss = self.adversarial_loss(self.discriminator(fake_images.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        return g_loss.item(), d_loss.item()
    
# Example usage
if __name__ == "__main__":
    srgan = SRGAN()
    high_res = torch.randn((1, 3, 256, 256))  # Simulated high-resolution image
    low_res = torch.randn((1, 3, 64, 64))  # Simulated low-resolution image
    g_loss, d_loss = srgan.train_step(high_res, low_res)
    print(f"Generator Loss: {g_loss}, Discriminator Loss: {d_loss}")