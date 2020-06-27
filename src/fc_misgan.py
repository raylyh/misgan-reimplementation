import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from generators_fc import fcDataGenerator, fcMaskGenerator
from discriminators_fc import fcDiscriminator
from load_data import squareMNIST, randomDropoutMNIST
from helper import plot_grid, mask_data

import matplotlib.pyplot as plt
import sys
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(sys.version)  # Python 3.6
print(device)

###################
# Hyperparameters #
###################
# our choice
length = 13
dropout_prob = 0.5

image_size = 28

data = squareMNIST(length=length, image_size=image_size)
# data = randomDropoutMNIST(p=dropout_prob, image_size=image_size)

# From DCGAN paper
batch_size = 128

# From MisGAN
epochs = 300
latent_size = 128
temperature = 0.66
alpha = 0.2

# From WGAN-GP
penalty_lambda = 10
n_critic = 5

# Adam
adam_alpha = 0.0001
adam_beta_1 = 0.5
adam_beta_2 = 0.9

###################################
# Put the dataset into DataLoader #
###################################
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)


# Create all the generators, discrimnators
G_data = fcDataGenerator(latent_size=latent_size, image_size=image_size, batch_size=batch_size).to(device)
G_mask = fcMaskGenerator(latent_size=latent_size, temperature=temperature, image_size=image_size, batch_size=batch_size).to(device)

D_data = fcDiscriminator(image_size=image_size, batch_size=batch_size).to(device)
D_mask = fcDiscriminator(image_size=image_size, batch_size=batch_size).to(device)


# Create all the optimisers
G_data_optimizer = optim.Adam(G_data.parameters(), lr=adam_alpha, betas=(adam_beta_1, adam_beta_2))
G_mask_optimizer = optim.Adam(G_mask.parameters(), lr=adam_alpha, betas=(adam_beta_1, adam_beta_2))

D_data_optimizer = optim.Adam(D_data.parameters(), lr=adam_alpha, betas=(adam_beta_1, adam_beta_2))
D_mask_optimizer = optim.Adam(D_mask.parameters(), lr=adam_alpha, betas=(adam_beta_1, adam_beta_2))


############################
# training initialisations #
############################

# Initialise the latent space array in the generator
data_noise = torch.empty(batch_size, latent_size, 1, 1, dtype=torch.float).to(device)
mask_noise = torch.empty(batch_size, latent_size, 1, 1, dtype=torch.float).to(device)

# Initialise the epsilon for data interpolation
epsilons = torch.empty(batch_size, 1, 1, 1, dtype=torch.float).to(device)

# Define the update function for WGAN with gradient penalty
def gradient_penalty(disc, real, fake):
  # generate interpolated data with epsilons
  epsilons.uniform_(0, 1)
  data_itp = (real + (1 - epsilons) * fake).clone().detach().requires_grad_(True).to(device)

  # Compute WGAN Loss with Gradient Penalty
  gradients = autograd.grad(outputs=disc(data_itp), inputs=data_itp,
                            grad_outputs=torch.ones(batch_size, 1).to(device), create_graph=True)[0]
  gradients = gradients.view(batch_size, -1)
  gradient_penalty = penalty_lambda * ((gradients.norm(dim=1) - 1)**2).mean()

  return gradient_penalty



#################
# training loop #
#################
plot_interval = 100
updates = 0

# update the discriminator for n_critic batches before updating the generator for one batch
for epoch in range(epochs):
  data_loader = tqdm(data_loader)
  for data_real, mask_real, _ in data_loader:

    # clear the gradient in Discriminators
    D_data_optimizer.zero_grad()
    D_mask_optimizer.zero_grad()
    
    # put real data to device
    data_real, mask_real = data_real.to(device), mask_real.to(device)
    # generate fake data
    data_noise.normal_(0, 1)
    mask_noise.normal_(0, 1)
    data_fake, mask_fake = G_data(data_noise), G_mask(mask_noise)
    # generate masked data
    data_masked_real = data_real * mask_real
    data_masked_fake = data_fake * mask_fake

    #################################################
    # Discriminators Loss: masked data and the mask #
    #################################################

    # data_masked: Compute WGAN Loss with Gradient Penalty
    penalty = gradient_penalty(D_data, data_masked_real.data, data_masked_fake.data)
    loss_data = D_data(data_masked_fake).mean() - D_data(data_masked_real).mean() + penalty
    loss_data.backward(retain_graph=True)
    D_data_optimizer.step()

    # mask: Compute WGAN Loss with Gradient Penalty
    penalty = gradient_penalty(D_mask, mask_real.data, mask_fake.data)
    loss_mask = D_mask(mask_fake).mean() - D_mask(mask_real).mean() + penalty
    loss_mask.backward(retain_graph=True)
    D_mask_optimizer.step()

    updates += 1
    #############################################
    # Generators Loss: masked data and the mask #
    #############################################

    if updates == n_critic:

      # clear the gradient in optimizers
      G_data_optimizer.zero_grad()
      G_mask_optimizer.zero_grad()

      # generate fake data
      data_noise.normal_(0, 1)
      mask_noise.normal_(0, 1)

      data_fake, mask_fake = G_data(data_noise), G_mask(mask_noise)
      # generate masked data
      data_masked_fake = data_fake * mask_fake

      # Compute the loss on G_data
      loss_data_G = -1 * D_data(data_masked_fake).mean()
      loss_data_G.backward(retain_graph=True)

      # Compute the loss on G_mask
      loss_mask_G = -1 * D_mask(mask_fake).mean()
      loss_mask_G += loss_data_G * alpha
      loss_mask_G.backward()

      G_data_optimizer.step()
      G_mask_optimizer.step()

      # reset the critic updates to 0
      updates = 0


  if epoch % plot_interval == 0:
      G_data.eval()
      G_mask.eval()

      with torch.no_grad():
          print('Epoch:', epoch + 1)
          fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
          
          data_noise.normal_()
          data_samples = G_data(data_noise)
          plot_grid(ax1, data_samples, image_size, title='generated complete data')
          
          mask_noise.normal_()
          mask_samples = G_mask(mask_noise)
          plot_grid(ax2, mask_samples, image_size, title='generated masks')
          
          plt.show()
          plt.close(fig)

      G_data.train()
      G_mask.train()

print("Done training")
# torch.save(G_data.state_dict(), "G_data_"+str(length))
# torch.save(G_mask.state_dict(), "G_mask_"+str(length))
# torch.save(D_data.state_dict(), "D_data_"+str(length))
# torch.save(D_mask.state_dict(), "D_mask_"+str(length))

