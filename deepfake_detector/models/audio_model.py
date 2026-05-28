# import torch
# import torch.nn as nn

# class AudioCNN(nn.Module):
#     def __init__(self):
#         super(AudioCNN, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(32 * 10 * 10, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)



import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Adaptive pooling fixes size mismatch automatically
        self.pool = nn.AdaptiveAvgPool2d((10, 10))

        self.fc = nn.Sequential(
            nn.Linear(32 * 10 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)   # 🔥 THIS FIXES YOUR ERROR
        x = x.view(x.size(0), -1)
        return self.fc(x)