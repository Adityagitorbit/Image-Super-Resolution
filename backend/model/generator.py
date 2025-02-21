import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A residual block used in the generator. Helps in stable training and better feature extraction.
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual  # Skip connection (x + residual) for stability


class Generator(nn.Module):
    """
    The Generator network for Super-Resolution.
    """

    def __init__(self, in_channels=3, num_residual_blocks=16):
        super(Generator, self).__init__()

        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.PReLU()

        # Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # Post Residual Convolution
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Upsampling Layers
        self.upsample1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=2)  # Upscale by 2x
        self.upsample2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=2)  # Upscale by 2x

        # Final Output Layer
        self.conv3 = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()  # Normalize output image pixels to [-1, 1]

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        residual = self.residual_blocks(out1)
        out2 = self.bn2(self.conv2(residual)) + out1  # Skip connection

        up1 = self.pixel_shuffle1(self.upsample1(out2))
        up2 = self.pixel_shuffle2(self.upsample2(up1))
        
        output = self.tanh(self.conv3(up2))
        return output


# Test the Generator Model
if __name__ == "__main__":
    gen = Generator()
    test_input = torch.randn((1, 3, 64, 64))  # Batch size 1, 3 channels, 64x64 image
    test_output = gen(test_input)
    print(f"Input Shape: {test_input.shape}")
    print(f"Output Shape: {test_output.shape}")  # Should be (1, 3, 256, 256)
