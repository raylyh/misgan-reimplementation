import torch.nn as nn

# Data Discriminator and Mask Discriminator (Critic)
class fcDiscriminator(nn.Module):
  def __init__(self, image_size, batch_size):
    super().__init__()
    self.batch_size = batch_size

    self.fc1 = nn.Sequential(
      nn.Linear(image_size * image_size, 512),
      nn.ReLU()
    )
    self.fc2 = nn.Sequential(
      nn.Linear(512, 256),
      nn.ReLU()
    )
    self.fc3 = nn.Sequential(
      nn.Linear(256, 128),
      nn.ReLU()
    )
    self.fc4 = nn.Sequential(
      nn.Linear(128, 1)
    )
    # linear activation for the last layer to predice the score of 'realness' given an image

  def forward(self, input):
    out = input.view(self.batch_size, -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    out = self.fc4(out)
    return out