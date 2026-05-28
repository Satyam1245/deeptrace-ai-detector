import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from deepfake_detector.models.audio_model import AudioCNN

# Dummy dataset (replace later with real dataset)
X = np.random.rand(100, 1, 40, 100)
y = np.random.randint(0, 2, 100)

X = torch.tensor(X).float()
y = torch.tensor(y).float().unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = AudioCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for inputs, labels in loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "outputs/checkpoints/audio_model.pth")

print("✅ Audio model saved!")