import torch
import torch.nn as nn

class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=5, padding=2),#####
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        in_feature = 221184
        self.fc1 = nn.Linear(in_features=in_feature, out_features=2)
        self.fc2 = nn.Linear(in_features=in_feature, out_features=7)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        sex = nn.functional.softmax(self.fc1(x), dim=1)
        age = nn.functional.softmax(self.fc2(x), dim=1)
        return sex, age