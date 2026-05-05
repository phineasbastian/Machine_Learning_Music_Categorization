

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GenreCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(GenreCNN, self).__init__()
#         # Initial layer: Extracts basic spectral textures (edges and tones)
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
        
#         # ASYMMETRIC PARALLEL BRANCHES: Decouples pitch and rhythm
#         # Vertical Kernel: Scans frequency stacks to identify Pitch/Harmonics
#         self.conv2_freq = nn.Conv2d(32, 64, kernel_size=(7, 1), padding=(3, 0))
#         # Horizontal Kernel: Scans the timeline to identify Rhythm/Beats
#         self.conv2_time = nn.Conv2d(32, 64, kernel_size=(1, 7), padding=(0, 3))
#         self.bn2 = nn.BatchNorm2d(128) 
        
#         # Deep Feature Layer: Combines simple patterns into complex genre identities
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
        
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.4) # Prevents memorization of training data
#         self.adaptive_pool = nn.AdaptiveMaxPool2d((4, 4)) # Forces consistent output size
#         self.fc1 = nn.Linear(256 * 4 * 4, 512)
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = x.unsqueeze(1) # Adds color channel dimension (Required for CNN)
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
#         # Parallel processing: Analyzing Pitch and Rhythm in tandem
#         x_f = self.conv2_freq(x)
#         x_t = self.conv2_time(x)
#         x = torch.cat((x_f, x_t), dim=1) # Merges musical features
#         x = self.pool(F.relu(self.bn2(x)))
        
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.adaptive_pool(x)
#         x = torch.flatten(x, 1) # Flattens image into a 1D vector for classification
#         x = self.dropout(F.relu(self.fc1(x)))
#         return self.fc2(x)





import torch
import torch.nn as nn
import torch.nn.functional as F


class GenreCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Stage 1: learn basic local spectrogram patterns
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Stage 2: parallel asymmetric branches
        # Frequency branch scans vertically for harmonic / pitch structure
        self.conv2_freq = nn.Conv2d(32, 64, kernel_size=(7, 1), padding=(3, 0))
        # Time branch scans horizontally for rhythm / transient structure
        self.conv2_time = nn.Conv2d(32, 64, kernel_size=(1, 7), padding=(0, 3))
        self.bn2 = nn.BatchNorm2d(128)

        # Stage 3: fuse both branches into higher-level genre features
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Shared utility layers
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((4, 4))

        # Classifier head
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input arrives as [batch, mel_bins, time]
        # Add channel dimension -> [batch, 1, mel_bins, time]
        x = x.unsqueeze(1)

        # Basic feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Parallel asymmetric analysis
        x_freq = self.conv2_freq(x)
        x_time = self.conv2_time(x)

        # Merge frequency and time features along channel dimension
        x = torch.cat((x_freq, x_time), dim=1)
        x = self.pool(F.relu(self.bn2(x)))

        # Deeper shared feature extraction
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Force fixed spatial size before classifier
        x = self.adaptive_pool(x)

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x