import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        padding = (dilation * (kernel_size - 1) + 1) // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode='replicate'
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class PositionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_block = nn.ModuleList([
            DenseCNNBlock(15, 8, 3),
            DenseCNNBlock(15+8, 20, 3),
            DenseCNNBlock(15+8+20, 8, 3),
            DenseCNNBlock(15+8+20+8, 8, 5, dilation=2)
        ])
        
        # Calculate total channels from all dense blocks + input
        self.total_channels = 15 + 8 + 20 + 8 + 8
        
        # Final 1x1 convolution to produce 5 output channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.total_channels, 5, kernel_size=1),
            nn.Sigmoid()  # Use Sigmoid for binary output per channel
        )
        
    def forward(self, x):
        features = [x]
        for layer in self.dense_block:
            concat_features = torch.cat(features, dim=1)
            new_features = layer(concat_features)
            features.append(new_features)
            
        # Concatenate all features and apply final convolution
        final_features = torch.cat(features, dim=1)
        return self.final_conv(final_features)