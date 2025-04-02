# GAN for FashionMNIST Image Generation

This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN) for generating FashionMNIST images. The model consists of a generator and discriminator trained in an adversarial manner, with evaluation using the Fréchet Inception Distance (FID) metric.

## Model Architecture

### Generator
The generator uses transposed convolutional layers to upsample from a latent space vector to a 28x28 FashionMNIST image:
- Input: 64-dimensional latent vector
- Architecture:
  - ConvTranspose2d (64→128, kernel=7)
  - BatchNorm + ReLU
  - ConvTranspose2d (128→64, kernel=4, stride=2)
  - BatchNorm + ReLU
  - ConvTranspose2d (64→1, kernel=4, stride=2)
  - Sigmoid activation

### Discriminator
The discriminator uses convolutional layers to classify images as real or fake:
- Input: 28x28 grayscale image
- Architecture:
  - Conv2d (1→64, kernel=4, stride=2)
  - LeakyReLU (0.2)
  - Conv2d (64→128, kernel=4, stride=2)
  - BatchNorm + LeakyReLU (0.2)
  - Flatten + Linear layer
  - Sigmoid activation

> Model Architectuer
![Discriminator Generator Arch](https://github.com/user-attachments/assets/561893b5-d7ef-49ac-9dec-672a20f724bf)


## Training Details

- **Dataset**: FashionMNIST (60,000 training images)
- **Batch size**: 720
- **Epochs**: 100
- **Optimizer**: Adam (lr=0.0002, β1=0.5, β2=0.999)
- **Loss function**: Binary Cross Entropy (BCE)
- **Training strategy**: Alternating updates for generator and discriminator
- **Evaluation metric**: Fréchet Inception Distance (FID)

## Results

The training achieved the following final metrics:
- Final Generator Loss: 0.9159
- Final Discriminator Loss: 1.1765
- Final FID Score: 70.5506

The loss curves show the adversarial training dynamics between the generator and discriminator.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- pytorch-fid (for FID calculation)
- tqdm (for progress bars)

## Usage

1. Install dependencies:
```bash
pip install torch torchvision pytorch-fid tqdm
```

2. Run the training script:
```python
python gan_fashionmnist.py
```

3. Generated images will be saved at specific epochs (1, 50, 100) as `generated_images_epoch_X.png`

## Notes

- The model uses smoothed labels (0.9 for real, 0 for fake) for more stable training
- FID is calculated by comparing 500 real images with 500 generated images
- Training takes approximately 8 seconds per epoch on GPU

## Sample Output

The training progress will display epoch information including:
- Epoch number
- Discriminator and generator losses
- FID score at evaluation checkpoints

> Sample output after 100 Epochs
![generated_images_epoch_100](https://github.com/user-attachments/assets/9edfeb37-e796-4c79-b8cb-f6ad33c7599a)

> Loss Plot
![loss-plot](https://github.com/user-attachments/assets/1ccbdf36-0869-4038-831d-78695a8b0da3)


Example output:
```
Epoch [100/100] | D Loss: 1.1765 | G Loss: 0.9159
FID Score at Epoch 100: 70.5506
```
