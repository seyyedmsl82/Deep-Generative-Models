"""
This module defines a Variational Autoencoder (VAE) with Encoder and Decoder classes to facilitate
unsupervised learning of compact representations of input data. The VAE model includes the capability
to sample from the learned latent space, compute loss metrics, and apply the reparameterization trick
to allow backpropagation through stochastic nodes. 

The module supports training with optional validation, visualization of loss metrics, and generation
of new samples from the latent distribution.
"""


import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


def sampling(mean, log_var):
    """
    Samples from the latent distribution using the reparameterization trick, allowing gradients to flow through
    stochastic nodes.

    :param mean: Tensor representing the mean of the latent distribution, shape (batch_size, latent_dim)
    :param log_var: Tensor representing the log variance of the latent distribution, shape (batch_size, latent_dim)
    :return: Sampled latent vector z, tensor of shape (batch_size, latent_dim)
    """
    epsilon = torch.randn_like(mean)
    z = mean + torch.exp(log_var * 0.5) * epsilon
    return z


def plot_loss(loss_list, name):
    """
    Plots the loss over the training iterations.

    :param loss_list: List of loss values to plot
    :param name: Name of the loss being plotted
    """
    plt.figure(figsize=(10, 5))
    plt.title(f"{name} During Training")
    plt.plot(loss_list, label=f"{name}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


class Encoder(nn.Module):
    """
    Encoder module for the VAE. Compresses input images into a latent space represented by mean and log variance.
    """

    def __init__(self, latent_dim=16):
        """
        Initializes the Encoder layers.

        :param latent_dim: Dimensionality of the latent space
        """
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
        self.fc1 = nn.Linear(4096, 2 * latent_dim)  # Not necessary
        self.bn5 = nn.BatchNorm1d(4096)
        self.mean = nn.Linear(4096, latent_dim)
        self.log_var = nn.Linear(4096, latent_dim)

    def forward(self, x):
        """
        Forward pass through the Encoder to obtain mean and log variance for the latent space.

        :param x: Input tensor of shape (batch_size, channels, height, width)
        :return: Mean and log variance tensors of shape (batch_size, latent_dim)
        """
        batch_size = x.size(0)
        x = self.relu(self.dropout(self.bn1(self.conv1(x))))
        x = self.relu(self.dropout(self.bn2(self.conv2(x))))
        x = self.relu(self.dropout(self.bn3(self.conv3(x))))
        x = self.relu(self.dropout(self.bn4(self.conv4(x))))
        x = x.view(batch_size, -1)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    """
    Decoder module for the VAE. Reconstructs images from the latent representation.
    """

    def __init__(self):
        """
        Initializes the Decoder layers.
        """
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(16, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass through the Decoder to reconstruct the input images from the latent vector.

        :param x: Latent representation tensor, shape (batch_size, latent_dim)
        :return: Reconstructed image tensor, shape (batch_size, channels, height, width)
        """
        batch_size = x.size(0)
        x = self.relu(self.fc1(x))
        x = x.view(batch_size, 64, 8, 8)
        x = self.relu(self.dropout(self.bn2(self.deconv1(x))))
        x = self.relu(self.dropout(self.bn3(self.deconv2(x))))
        x = self.relu(self.dropout(self.bn4(self.deconv3(x))))
        x = self.tanh(self.deconv4(x))
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) combining Encoder and Decoder modules to learn a latent representation of input data.
    """

    def __init__(self, encoder, decoder, z_dim=2):
        """
        Initializes the VAE with an encoder, decoder, and latent dimension.

        :param encoder: Encoder module
        :param decoder: Decoder module
        :param z_dim: Dimensionality of the latent space
        """
        super(VAE, self).__init__()
        
        ###########################################################################################################
        # These parameters are not essential during training process, but they are necessary to load the model, 
        # since they are included in structure.
        # START:
        self.z_dim = z_dim

        self.enc = Encoder()
        self.dec = Decoder()

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        # END
        ###########################################################################################################

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
        Computes the VAE loss, combining reconstruction and KL divergence losses.

        :param reconstruction: Reconstructed images, shape (batch_size, channels, height, width)
        :param x: Original images, shape (batch_size, channels, height, width)
        :param mean: Latent mean tensor, shape (batch_size, latent_dim)
        :param log_var: Latent log variance tensor, shape (batch_size, latent_dim)
        :return: Total loss, KL divergence, and reconstruction loss
        """
        recon_loss = 5 * ((reconstruction - x) ** 2).mean()
        kl_div = 0.1 * (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0))
        total_loss = recon_loss + kl_div
        return total_loss, kl_div, recon_loss

    def forward(self, x):
        """
        Forward pass through the VAE. Encodes the input, samples from the latent distribution, and decodes.

        :param x: Input images, shape (batch_size, channels, height, width)
        :return: Reconstructed images, mean, and log variance tensors
        """
        mean, log_var = self.encoder(x)
        z = sampling(mean, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_var

    def train_vae(self, train_dataloader, val_dataloader=None, epochs=1000, lr=0.0005, device='cuda'):
        """
        Trains the VAE, with optional validation.

        :param train_dataloader: DataLoader for training data
        :param val_dataloader: DataLoader for validation data (optional)
        :param epochs: Number of training epochs
        :param lr: Learning rate for the optimizer
        :param device: Device to use for training (e.g., 'cuda' or 'cpu')
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            train_dataloader_with_progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            total_epoch_loss, kl_epoch_loss, recon_epoch_loss = 0, 0, 0
            self.train()
            for data in train_dataloader_with_progress:
                data = data.to(device)
                optimizer.zero_grad()
                reconstruction, mean, log_var = self.forward(data)
                loss, kl_loss, recon_loss = self.loss_function(reconstruction, data, mean, log_var)
                loss.backward()
                optimizer.step()
                total_epoch_loss += loss.item()
                kl_epoch_loss += kl_loss.item()
                recon_epoch_loss += recon_loss.item()
            self.epoch_loss_list.append(total_epoch_loss)
            self.kl_loss_list.append(kl_epoch_loss)
            self.recon_loss_list.append(recon_epoch_loss)
            print(f"Training - Epoch [{epoch+1}/{epochs}], Total Loss: {total_epoch_loss:.4f}, KL Loss: {kl_epoch_loss:.4f}, Recon Loss: {recon_epoch_loss:.4f}")
            if val_dataloader:
                self.eval()
                val_total_epoch_loss, val_kl_epoch_loss, val_recon_epoch_loss = 0, 0, 0
                with torch.no_grad():
                    for data in val_dataloader:
                        data = data.to(device)
                        reconstruction, mean, log_var = self.forward(data)
                        loss, kl_loss, recon_loss = self.loss_function(reconstruction, data, mean, log_var)
                        val_total_epoch_loss += loss.item()
                        val_kl_epoch_loss += kl_loss.item()
                        val_recon_epoch_loss += recon_loss.item()
                self.val_epoch_loss_list.append(val_total_epoch_loss)
                self.val_kl_loss_list.append(val_kl_epoch_loss)
                self.val_recon_loss_list.append(val_recon_epoch_loss)
                print(f"Validation - Epoch [{epoch+1}/{epochs}], Total Loss: {val_total_epoch_loss:.4f}, KL Loss: {val_kl_epoch_loss:.4f}, Recon Loss: {val_recon_epoch_loss:.4f}")
        plot_loss(self.epoch_loss_list, "Training Total Loss")
        plot_loss(self.kl_loss_list, "Training KL Loss")
        plot_loss(self.recon_loss_list, "Training Reconstruction Loss")
        if val_dataloader:
            plot_loss(self.val_epoch_loss_list, "Validation Total Loss")
            plot_loss(self.val_kl_loss_list, "Validation KL Loss")
            plot_loss(self.val_recon_loss_list, "Validation Reconstruction Loss")

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
