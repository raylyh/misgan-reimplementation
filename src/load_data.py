import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

import random

# Square Observation
class squareMNIST(Dataset):
  def __init__(self, length, image_size):
    super().__init__()

    transform = transforms.Compose([
      transforms.Resize((image_size, image_size)),   # resize the image to the desired image size
      transforms.ToTensor(),  # transform an image to [0, 1] in the shape of (C, H, W)
    ])
    mnist = MNIST('./data', train=True, transform=transform, download=True)

    self.size = len(mnist)
    self.length = length
    self.data = torch.empty(self.size, 1, image_size, image_size, dtype=torch.float)
    self.mask = torch.zeros(self.size, 1, image_size, image_size, dtype=torch.float)
    self.mask_loc = torch.empty(self.size, 2, dtype=torch.uint8)

    for i, (img, label) in enumerate(mnist):
      # apply the random square mask
      x = random.randint(0, image_size - length)
      y = random.randint(0, image_size - length)
      self.mask[i][0][x:x+length, y:y+length] = 1.0
      # save the original image
      self.data[i][0] = img
      # save the location of the mask
      self.mask_loc[i, :] = torch.tensor([x, y])


  def __getitem__(self, i):
    # original, mask, index
    return self.data[i], self.mask[i], i


  def __len__(self):
    return self.size




# Independent Dropout from a Bernoulli Distribution
class randomDropoutMNIST(Dataset):
  def __init__(self, p, image_size):
    super().__init__()

    transform = transforms.Compose([
      transforms.Resize((image_size, image_size)),   # resize the image to the desired image size
      transforms.ToTensor(),  # transform an image to [0, 1] in the shape of (C, H, W)
    ])
    mnist = MNIST('./data', train=True, transform=transform, download=True)

    self.size = len(mnist)
    self.prob = p
    self.data = torch.empty(self.size, 1, image_size, image_size, dtype=torch.float)
    self.mask = torch.zeros(self.size, 1, image_size, image_size, dtype=torch.float)

    for i, (img, label) in enumerate(mnist):
      # apply the random dropout to the mask
      self.mask[i][0].bernoulli_(p)
      # save the original image
      self.data[i][0] = img
  

  def __getitem__(self, i):
    # original, mask, index
    return self.data[i], self.mask[i], i


  def __len__(self):
    return self.size


