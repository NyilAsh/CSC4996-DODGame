import torch.nn as nn
import torch.nn.functional as F

class PositionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: (batch_size, 15, 10, 10)
        
        # First convolution block
        self.conv1 = nn.Conv2d(15, 8, kernel_size=2, padding=1)  # Output: (8, 11, 11)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(8, 20, kernel_size=3, padding=1)  # Output: (20, 11, 11)
        self.bn2 = nn.BatchNorm2d(20)
        
        # Third convolution block
        self.conv3 = nn.Conv2d(20, 8, kernel_size=4, padding=2)  # Output: (8, 12, 12)
        self.bn3 = nn.BatchNorm2d(8)
        
        # Dilated convolution block
        self.conv4 = nn.Conv2d(8, 8, kernel_size=5, dilation=2, padding=4)  # Output: (8, 12, 12)
        self.bn4 = nn.BatchNorm2d(8)
        
        # Final 1x1 convolution to get 5 output channels
        self.conv_final = nn.Conv2d(8, 5, kernel_size=1)  # Output: (5, 12, 12)
        
        # Upsample to match original size if needed
        self.upsample = nn.Upsample(size=(10, 10), mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Dilated conv block
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Final 1x1 conv
        x = self.conv_final(x)
        
        # Upsample if needed (remove if you want 12x12 output)
        x = self.upsample(x)
        
        return x