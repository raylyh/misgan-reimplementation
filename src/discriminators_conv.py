import torch.nn as nn

# Data Discriminator and Mask Discriminator (Critic)
# LayerNorm for WGAN-GP or no BatchNorm at all
class convDiscriminator(nn.Module):
  def __init__(self, features, batch_size):
    super().__init__()
    self.batch_size = batch_size

    # input is 1 x 32 x 32
    self.conv1 = nn.Sequential(
      nn.Conv2d(1, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.2)
    )
    # 128 x 16 x 16
    self.conv2 = nn.Sequential(
      nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
      nn.LayerNorm([features * 4, 8, 8]),
      nn.LeakyReLU(0.2)
    )
    # 256 x 8 x 8
    self.conv3 = nn.Sequential(
      nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1, bias=False),
      nn.LayerNorm([features * 8, 4, 4]),
      nn.LeakyReLU(0.2)
    )
    # 512 x 4 x 4
    # linear activation for the last layer to predice the score of 'realness' given an image
    # so no sigmoid
    self.fc = nn.Linear(features * 8 * 4 * 4, 1)

  def forward(self, input):
    out = self.conv1(input)
    out = self.conv2(out)
    out = self.conv3(out)
    out = out.view(self.batch_size, -1)
    out = self.fc(out)
    return out