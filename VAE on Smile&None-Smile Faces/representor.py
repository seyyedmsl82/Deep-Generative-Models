"""
This script contains functions for working with a trained Variational Autoencoder (VAE) model, specifically for 
loading the model, displaying samples, and adjusting specific attributes (e.g., "smile" intensity) by manipulating 
the latent space. It also includes utility functions for calculating average latent vectors and obtaining latent 
representations of individual images.

Functions:
    - load_model: Loads a VAE model with specified encoder and decoder classes from a checkpoint.
    - show_samples: Displays a grid of sample images.
    - encoded_real_samples: Encodes and decodes real samples to display reconstructed outputs.
    - calculate_average_latent_vector: Calculates the average latent vector for a given dataset.
    - get_latent_vector_for_single_image: Gets the latent vector of a single image.
    - image_smile_adjustment: Adjusts a latent vector to simulate a smile by modifying specific dimensions.
"""

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from vae import Encoder, Decoder, VAE
from dataloader import FaceDataSet
import numpy as np
from PIL import Image


def load_model(encoder_class, decoder_class, path, device='cuda'):
    """
    Loads a trained VAE model with specified encoder and decoder classes from a given checkpoint path.

    :param encoder_class: The Encoder class to initialize
    :param decoder_class: The Decoder class to initialize
    :param path: Path to the saved model checkpoint
    :param device: Device to load the model onto ('cuda' or 'cpu')
    :return: Loaded VAE model
    """
    checkpoint = torch.load(path, map_location=device)
    encoder = encoder_class
    decoder = decoder_class
    model = VAE(encoder, decoder).to(device)

    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.load_state_dict(checkpoint['vae_state_dict'])
    print(f"Model loaded from {path}")
    return model


def show_samples(samples, nrows=2, ncols=5):
    """
    Displays a grid of images from sample tensors.

    :param samples: Tensor of images to display, shape (num_samples, channels, height, width)
    :param nrows: Number of rows in the grid
    :param ncols: Number of columns in the grid
    """
    samples = samples.cpu().numpy()
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(len(samples)):
        axes[i].imshow(samples[i].transpose(1, 2, 0) * 0.5 + 0.5)
        axes[i].axis('off')

    plt.show()


def encoded_real_samples(model, device='cuda', batch_size=128):
    """
    Encodes real samples from a dataset and displays the decoded output.

    :param model: VAE model to encode and decode the samples
    :param device: Device to use for processing ('cuda' or 'cpu')
    :param batch_size: Number of samples to process in a batch
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = FaceDataSet('./train/total', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    data_iter = iter(dataloader)
    samples = next(data_iter)
    samples = samples[:36]
    samples = samples.to(device)

    model.eval()
    with torch.no_grad():
        means, log_vars = model.encoder(samples)
        decoded_images = model.decoder(means)

    show_samples(decoded_images, nrows=6, ncols=6)


def calculate_average_latent_vector(model, data_path, device='cuda'):
    """
    Calculates the average latent vector of images from a dataset path.

    :param model: VAE model to encode the images
    :param data_path: Path to the dataset for averaging
    :param device: Device to use for processing ('cuda' or 'cpu')
    :return: Average latent vector tensor
    """
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = FaceDataSet(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    latent_sum = torch.zeros(model.encoder.latent_dim).to(device)
    num_images = 0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            mean, _ = model.encoder(data)
            latent_sum += mean.squeeze(0)
            num_images += 1

    avg_latent_vector = latent_sum / num_images
    print(f"Calculated average latent vector for {data_path}")
    return avg_latent_vector


def get_latent_vector_for_single_image(model, image_path, device='cuda'):
    """
    Gets the latent vector representation of a single image.

    :param model: VAE model to encode the image
    :param image_path: Path to the image file
    :param device: Device to use for processing ('cuda' or 'cpu')
    :return: Tuple of mean and log variance tensors for the latent representation
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        mean, log_var = model.encoder(image)
    return mean.squeeze(0), log_var.squeeze(0)


def image_smile_adjustment(image_path, dimensions, smile_strength, model):
    """
    Adjusts a latent vector to simulate a smile by modifying specific latent dimensions.

    :param image_path: Path to the input image
    :param dimensions: Indices of latent dimensions associated with a smile
    :param smile_strength: Tensor of values representing strength of smile-related changes
    :param model: VAE model to encode and decode the image
    """
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()
    mean, log_var = get_latent_vector_for_single_image(model, image_path)
    samples = []

    for _ in range(9):
        for j, dim in enumerate(dimensions[:5]):
            if j in [2, 3, 4]:
                continue
            mean[dim] -= smile_strength[j] / 1.5

        model.eval()
        with torch.no_grad():
            sample = model.decoder(mean.unsqueeze(0))
            samples.append(sample.squeeze(0).cpu())

    samples = torch.stack(samples)
    show_samples(samples, 3, 3)


# Load the trained VAE model
vae = load_model(Encoder(), Decoder(), 'vae_test.pth')

# Display encoded real samples
encoded_real_samples(vae)

# Display randomly generated samples from the VAE
images = vae.sample(36)
show_samples(images, 6, 6)

# Calculate average latent vectors for "smile" and "non-smile" images
smile_avg_latent = calculate_average_latent_vector(vae, 'train/smile')
non_smile_avg_latent = calculate_average_latent_vector(vae, 'train/non_smile')

# Calculate the difference to find the smile dimensions
latent_difference = (smile_avg_latent - non_smile_avg_latent).cpu()

# Get the top 10 values and their indices
top_values, top_indices = torch.topk(latent_difference.abs(), 5)
print("Top 10 differences and their indices (smile dimensions):")
for i in range(5):
    top_values[i] = smile_avg_latent[top_indices[i]].cpu() - non_smile_avg_latent[top_indices[i]].cpu()
    print(f"Index: {top_indices[i]}, Difference: {top_values[i]}")

# Adjust smile in an example image
img_path1 = 'train/smile/Jennifer_Aniston_0002.jpg'
image_smile_adjustment(img_path1, top_indices, top_values, vae)
