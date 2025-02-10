"""
This script trains a Variational Autoencoder (VAE) model for unsupervised learning on face images.
It includes data preprocessing, model training, and saving, with options to visualize generated samples.

Modules and libraries utilized:
- PyTorch and Torchvision for model definition, optimization, and transformations
- Sklearn for splitting data into training and validation sets
- Custom modules 'FaceDataSet', 'VAE', 'Encoder', 'Decoder', and 'show_samples' for dataset handling,
  model architecture, and sample visualization.
"""


import os
import torch
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataloader import FaceDataSet
from vae import VAE, Encoder, Decoder
from representor import show_samples


# Path to directory containing face images
main_path = 'train/total'

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Batch size for data loading
batch_size = 128


def save_model(model, path='vae_model.pth'):
    """
    Saves the VAE model, including encoder, decoder, and other parameters.

    :param model: VAE model to save
    :param path: Path to save the model parameters
    """
    # Save model state dictionaries for encoder, decoder, and full VAE
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'vae_state_dict': model.state_dict(),
        'latent_dim': model.encoder.latent_dim
    }, path)
    print(f"Model saved to {path}")


# Define image transformations for pre-processing
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

# Load dataset and split into training and validation sets
trainset = FaceDataSet(main_path, transform=transform)
trainset, valset = train_test_split(trainset, test_size=0.2, random_state=42)

# Create DataLoaders for training and validation sets
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

# Sample a few images to visualize
data_iter = iter(trainloader)
images = next(data_iter)
random_images = images[:9]
show_samples(random_images, nrows=3, ncols=3)  # Display 3x3 grid of sample images

# Initialize Encoder, Decoder, and VAE model, then move to device
encoder = Encoder().to(device)
decoder = Decoder().to(device)
vae = VAE(encoder, decoder).to(device)

# Train the VAE model
vae.train_vae(trainloader, valloader)

# Save the trained VAE model
save_model(vae)

# Generate and display samples from the VAE
samples = vae.sample(9)  # Generate 9 samples
show_samples(samples, 3, 3)  # Display generated samples in 3x3 grid
