import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from vae import Encoder, Decoder, VAE
from dataloader import FaceDataSet
import numpy as np
from PIL import Image


def load_model(encoder_class, decoder_class, path, device='cuda'):
    """

    :param encoder_class:
    :param decoder_class:
    :param path:
    :param device:
    :return:
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

    :param samples:
    :param nrows:
    :param ncols:
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

    :param model:
    """
    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset
    dataset = FaceDataSet('./train/total', gp='train', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get a batch of images
    data_iter = iter(dataloader)
    images = next(data_iter)  # Get one batch of images
    images = images[:36]
    images = images.to(device)

    # Decode the images
    model.eval()
    with torch.no_grad():
        means, log_vars = model.encoder(images)  # Get latent vectors
        decoded_images = model.decoder(means)  # Decode the latent vectors

    # Display the decoded images
    show_samples(decoded_images, nrows=6, ncols=6)


def calculate_average_latent_vector(model, data_path, device='cuda'):
    """
    Calculates the average latent vector of all images in a specified folder.
    """
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = FaceDataSet(data_path, gp='train', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    latent_sum = torch.zeros(model.encoder.latent_dim).to(device)
    num_images = 0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            mean, _ = model.encoder(data)
            latent_sum += mean.squeeze(0)  # Add the latent vector for this image
            num_images += 1

    avg_latent_vector = latent_sum / num_images
    print(f"Calculated average latent vector for {data_path}")
    return avg_latent_vector


def get_latent_vector_for_single_image(model, image_path, device='cuda'):
    """
    Gets the latent vector of a single image.
    """
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        mean, log_var = model.encoder(image)
    return mean.squeeze(0), log_var.squeeze(0)


def image_smile_adjustment(image_path, dimentions, smile_strength, model):
    """

    :param image_path:
    :param dimentions:
    :param smile_strength:
    :param model:
    """
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()
    mean, log_var = get_latent_vector_for_single_image(model, image_path)
    samples = []

    for _ in range(9):
        for j, dim in enumerate(dimentions[:5]):
            if j in [0, 1, 2, 4, 5]:
                continue
            mean[dim] -= smile_strength[j] / 1.2

        model.eval()
        with torch.no_grad():
            sample = model.decoder(mean.unsqueeze(0))
            samples.append(sample.squeeze(0).cpu())

    samples = torch.stack(samples)
    show_samples(samples, 3, 3)


# Load the trained VAE model
vae = load_model(Encoder(), Decoder(), 'vae_model_b;dlfkds;lkfs;.pth')
#
# Calculate average latent vectors for "smile" and "non-smile" images
smile_avg_latent = calculate_average_latent_vector(vae, 'train/smile')
non_smile_avg_latent = calculate_average_latent_vector(vae, 'train/non_smile')

# Calculate the difference to find the smile dimensions
latent_difference = (smile_avg_latent - non_smile_avg_latent).cpu()

# Get the top 10 values and their indices
top_values, top_indices = torch.topk(latent_difference.abs(), 10)
print("Top 10 differences and their indices (smile dimensions):")
for i in range(10):
    top_values[i] = smile_avg_latent[top_indices[i]].cpu() - non_smile_avg_latent[top_indices[i]].cpu()
    print(smile_avg_latent[top_indices[i]].cpu(), non_smile_avg_latent[top_indices[i]].cpu())
    print(f"Index: {top_indices[i]}, Difference: {top_values[i]}")

img_path = 'train/smile/Queen_Elizabeth_II_0001.jpg'
img_path1 = 'train/smile/Jennifer_Aniston_0002.jpg'
image_smile_adjustment(img_path1, top_indices, top_values, vae)

encoded_real_samples(vae)

samples = vae.sample(36)
show_samples(samples, 6, 6)

# model = load_model(Encoder(), Decoder(), 'vae_model_beta64.pt')
# samples = model.sample(9)
# show_samples(samples, 3, 3)
