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

main_path = 'train/total'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128


def save_model(model, path='vae_model_b;dlfkds;lkfs;.pth'):
    """
    Saves the VAE model, including encoder, decoder, and other parameters.

    :param model: VAE model to save
    :param path: Path to save the model parameters
    """
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'vae_state_dict': model.state_dict(),
        'latent_dim': model.encoder.latent_dim
    }, path)
    print(f"Model saved to {path}")


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = FaceDataSet(main_path, gp='train', transform=transform)
trainset, valset = (train_test_split(trainset,
                                     test_size=0.2,
                                     random_state=42)
                    )
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

# testset = FaceDataSet(main_path, gp='test', transform=transform)
# testloader = DataLoader(testset)


data_iter = iter(trainloader)
images = next(data_iter)
random_images = images[:9]
show_samples(random_images, nrows=3, ncols=3)

encoder = Encoder().to(device)
decoder = Decoder().to(device)
vae = VAE(encoder, decoder).to(device)

vae.train_vae(trainloader, valloader, epochs=1000)
save_model(vae)

samples = vae.sample(9)
show_samples(samples, 3, 3)
