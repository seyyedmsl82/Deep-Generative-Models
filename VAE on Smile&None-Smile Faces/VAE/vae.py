"""
This module implements a Variational Autoencoder (VAE) with an Encoder, Decoder, and Kullback-Leibler divergence loss
function.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    """
    Encoder part of the VAE, which compresses input images into a latent space represented by mean and log variance.
    """

    def __init__(self, latent_dim=64):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(4096, 2 * latent_dim)
        self.bn5 = nn.BatchNorm1d(4096)
        self.mean = nn.Linear(4096, latent_dim)
        self.log_var = nn.Linear(4096, latent_dim)

    def forward(self, x):
        """
        Forward pass through the Encoder. Processes the input through a series of convolutional, batch normalization,
        and ReLU activation layers, followed by a fully connected layer, producing the latent mean and log variance.

        :param x: Input tensor, a batch of images with shape (batch_size, 1, height, width)
        :return: mean and log_var tensors for the latent distribution, each of shape (batch_size, latent_dim)
        """
        batch_size = x.size()[0]
        x = self.relu(self.dropout(self.bn1(self.conv1(x))))
        x = self.relu(self.dropout(self.bn2(self.conv2(x))))
        x = self.relu(self.dropout(self.bn3(self.conv3(x))))
        x = self.relu(self.dropout(self.bn4(self.conv4(x))))
        x = x.view(batch_size, -1)
        # x = self.relu(self.fc1(x))
        #
        # mean, log_var = gaussian_parameters(x, dim=1)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    """
    Decoder part of the VAE, which reconstructs input images from the latent space representation.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(64, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2,
                                          output_padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass through the Decoder. Expands the latent representation back into the image dimensions
        through a series of transposed convolutions, batch normalization, and ReLU activations.

        :param x: Latent representation tensor, of shape (batch_size, latent_dim)
        :return: Reconstructed image tensor, of shape (batch_size, 1, height, width)
        """
        batch_size = x.size(0)
        x = self.relu(self.fc1(x))
        x = x.view(batch_size, 64, 8, 8)
        x = self.relu(self.dropout(self.bn2(self.deconv1(x))))
        x = self.relu(self.dropout(self.bn3(self.deconv2(x))))
        x = self.relu(self.dropout(self.bn4(self.deconv3(x))))
        x = self.tanh(self.deconv4(x))
        return x


def plot_loss(loss_list, name):
    """

    :param name:
    :param loss_list:
    """
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(loss_list, label=f"{name} Loss")

    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


class VAE(nn.Module):
    """
    Variational Autoencoder, which combines the Encoder and Decoder networks to learn a latent space representation
    of the input data.
    """

    def __init__(self, encoder, decoder, z_dim=2):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.recon_loss_list = []
        self.kl_loss_list = []
        self.epoch_loss_list = []
        self.val_recon_loss_list = []
        self.val_kl_loss_list = []
        self.val_epoch_loss_list = []

    def loss_function(self, reconstruction, x, mean, log_var):
        """
        Calculates the VAE loss, which is a combination of the reconstruction loss and KL divergence.

        :param reconstruction: Reconstructed images from the Decoder, tensor of shape (batch_size, 1, height, width)
        :param x: Original input images, tensor of shape (batch_size, 1, height, width)
        :param mean: Mean tensor from the Encoder's latent space, shape (batch_size, latent_dim)
        :param log_var: Log variance tensor from the Encoder's latent space, shape (batch_size, latent_dim)
        :return: Total VAE loss, a scalar tensor
        """
        recon_loss = torch.nn.MSELoss(reduction='sum')(reconstruction, x)
        # recon_loss = 5 * loss(reconstruction, x)
        # recon_loss = 5 * ((reconstruction - x) ** 2).mean()
        kl_div = 8 * (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()))
        total_loss = recon_loss + kl_div

        return total_loss, kl_div, recon_loss

    def sampling(self, mean, log_var):
        """
        Samples from the latent distribution using the reparameterization trick, which allows gradients to flow
        through stochastic nodes.

        :param mean: Mean of the latent distribution, tensor of shape (batch_size, latent_dim)
        :param log_var: Log variance of the latent distribution, tensor of shape (batch_size, latent_dim)
        :return: Sampled latent vector z, tensor of shape (batch_size, latent_dim)
        """
        epsilon = torch.randn_like(mean)
        z = mean + torch.exp(log_var * 0.5) * epsilon
        return z

    def forward(self, x):
        """
        Forward pass through the VAE. Encodes the input into a latent distribution, samples from it, and then decodes.

        :param x: Input tensor, a batch of images with shape (batch_size, 1, height, width)
        :return: Tuple containing the reconstructed images, the mean, and log variance of the latent distribution
        """
        mean, log_var = self.encoder(x)
        z = self.sampling(mean, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_var

    def train_vae(self, train_dataloader, val_dataloader=None, epochs=1000, lr=0.0005, device='cuda'):
        """
        Training loop for the VAE with optional validation dataset handling.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            train_dataloader_with_progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            total_epoch_loss = 0
            kl_epoch_loss = 0
            recon_epoch_loss = 0

            # Training loop
            self.train()
            for data in train_dataloader_with_progress:
                data = data.to(device)
                optimizer.zero_grad()
                reconstruction, mean, log_var = self.forward(data)
                loss, kl_loss, recon_loss = self.loss_function(reconstruction, data, mean, log_var)
                loss.backward()
                optimizer.step()

                # Accumulate training losses
                total_epoch_loss += loss.item()
                kl_epoch_loss += kl_loss.item()
                recon_epoch_loss += recon_loss.item()

            # Average training losses for the epoch
            avg_epoch_loss = total_epoch_loss / len(train_dataloader)
            avg_kl_loss = kl_epoch_loss / len(train_dataloader)
            avg_recon_loss = recon_epoch_loss / len(train_dataloader)

            print(
                f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f}, KL: {avg_kl_loss:.6f}, Recon: {avg_recon_loss:.4f}")

            self.recon_loss_list.append(avg_recon_loss)
            self.kl_loss_list.append(avg_kl_loss)
            self.epoch_loss_list.append(avg_epoch_loss)

            # Validation loop (if provided)
            if val_dataloader:
                val_total_loss, val_kl_loss, val_recon_loss = 0, 0, 0
                self.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    for val_data in val_dataloader:
                        val_data = val_data.to(device)
                        reconstruction, mean, log_var = self.forward(val_data)
                        val_loss, val_kl, val_recon = self.loss_function(reconstruction, val_data, mean, log_var)
                        val_total_loss += val_loss.item()
                        val_kl_loss += val_kl.item()
                        val_recon_loss += val_recon.item()

                # Average validation losses for the epoch
                avg_val_loss = val_total_loss / len(val_dataloader)
                avg_val_kl_loss = val_kl_loss / len(val_dataloader)
                avg_val_recon_loss = val_recon_loss / len(val_dataloader)

                print(
                    f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, KL: {avg_val_kl_loss:.6f}, Recon: {avg_val_recon_loss:.4f}")

                self.val_recon_loss_list.append(avg_val_recon_loss)
                self.val_kl_loss_list.append(avg_val_kl_loss)
                self.val_epoch_loss_list.append(avg_val_loss)

        plot_loss(self.recon_loss_list, 'Train Reconstruction')
        plot_loss(self.kl_loss_list, 'Train KL')
        plot_loss(self.epoch_loss_list, 'Train Epoch')
        if val_dataloader:
            plot_loss(self.val_recon_loss_list, 'Validation Reconstruction')
            plot_loss(self.val_kl_loss_list, 'Validation KL')
            plot_loss(self.val_epoch_loss_list, 'Validation Epoch')

    def sample(self, num_samples):
        """
        Generate samples from the VAE.

        :param num_samples: Number of samples to generate.
        :return: Generated images.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.encoder.latent_dim).to(next(self.parameters()).device)
            generated_images = self.decoder(z)
            return generated_images
