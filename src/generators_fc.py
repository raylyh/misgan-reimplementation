import torch
import torch.nn as nn

# Data Generator
class fcDataGenerator(nn.Module):
  def __init__(self, latent_size, image_size, batch_size):
    super().__init__()
    self.image_size = image_size
    self.batch_size = batch_size

    self.fc1 = nn.Sequential(
      nn.Linear(latent_size, 256),
      nn.ReLU()
    )
    self.fc2 = nn.Sequential(
      nn.Linear(256, 512),
      nn.ReLU()
    )
    self.fc3 = nn.Sequential(
      nn.Linear(512, image_size * image_size)
    )

  def forward(self, input):
    out = input.view(self.batch_size, -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    # use sigmoid as the data is scaled to [0, 1] (as suggested in MisGAN paper)
    out = torch.sigmoid(out)
    out = out.view(self.batch_size, 1, self.image_size, self.image_size)
    return out


# Mask Generator
class fcMaskGenerator(nn.Module):
  def __init__(self, latent_size, temperature, image_size, batch_size):
    super().__init__()
    self.image_size = image_size
    self.batch_size = batch_size
    self.temperature = temperature

    self.fc1 = nn.Sequential(
      nn.Linear(latent_size, 256),
      nn.ReLU()
    )
    self.fc2 = nn.Sequential(
      nn.Linear(256, 512),
      nn.ReLU()
    )
    self.fc3 = nn.Sequential(
      nn.Linear(512, image_size * image_size)
    )

  def forward(self, input):
    out = input.view(self.batch_size, -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    # use sigmoid as the data is scaled to [0, 1] (as suggested in MisGAN paper)
    out = torch.sigmoid(out / self.temperature)
    out = out.view(self.batch_size, 1, self.image_size, self.image_size)
    return out