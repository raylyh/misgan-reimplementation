import torch
import torch.nn as nn

# Data Generator
class convDataGenerator(nn.Module):
  def __init__(self, latent_size, features):
    super().__init__()

    # input is latent_size
    self.project = nn.Sequential(
      nn.ConvTranspose2d(latent_size, features * 8, kernel_size=4, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(features * 8),
      nn.ReLU()
    )
    # 512 x 4 x 4
    self.tconv1 = nn.Sequential(
      nn.ConvTranspose2d(features * 8, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(features * 4),
      nn.ReLU()
    )
    # 256 x 8 x 8
    self.tconv2 = nn.Sequential(
      nn.ConvTranspose2d(features * 4, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(features * 2),
      nn.ReLU()
    )
    # 128 x 16 x 16
    self.tconv3 = nn.ConvTranspose2d(features * 2, 1, kernel_size=4, stride=2, padding=1, bias=False)
    # 1 x 32 x 32

  def forward(self, input):
    out = self.project(input)
    out = self.tconv1(out)
    out = self.tconv2(out)
    out = self.tconv3(out)
    # use sigmoid as the data is scaled to [0, 1] (as suggested in MisGAN paper)
    out = torch.sigmoid(out)
    return out


# Mask Generator
class convMaskGenerator(nn.Module):
  def __init__(self, latent_size, features, temperature):
    super().__init__()
    self.temperature = temperature

    # input is latent_size
    self.project = nn.Sequential(
      nn.ConvTranspose2d(latent_size, features * 8, kernel_size=4, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(features * 8),
      nn.ReLU()
    )
    # 512 x 4 x 4
    self.tconv1 = nn.Sequential(
      nn.ConvTranspose2d(features * 8, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(features * 4),
      nn.ReLU()
    )
    # 256 x 8 x 8
    self.tconv2 = nn.Sequential(
      nn.ConvTranspose2d(features * 4, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(features * 2),
      nn.ReLU()
    )
    # 128 x 16 x 16
    self.tconv3 = nn.ConvTranspose2d(features * 2, 1, kernel_size=4, stride=2, padding=1, bias=False)
    # 1 x 32 x 32

  def forward(self, input):
    out = self.project(input)
    out = self.tconv1(out)
    out = self.tconv2(out)
    out = self.tconv3(out)
    # use sigmoid with temperature as the data is scaled to [0, 1] (as suggested in paper)
    out = torch.sigmoid(out / self.temperature)
    return out