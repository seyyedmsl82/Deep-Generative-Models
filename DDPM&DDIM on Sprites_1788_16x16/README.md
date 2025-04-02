# Diffusion Models Implementation

This repository contains a PyTorch implementation of Diffusion Models for image generation, focusing on Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM). The code is designed to work with the Sprites dataset (16x16 images) and includes training, sampling, and visualization utilities.

## Introduction

Diffusion models are a class of generative models that learn to generate data by gradually denoising a normally distributed variable. This implementation covers:

- **DDPM (Denoising Diffusion Probabilistic Models)**: The original formulation that uses a fixed Markov chain to gradually add and remove noise.
- **DDIM (Denoising Diffusion Implicit Models)**: An accelerated sampling method that allows faster generation while maintaining quality.

![Diffusion Process](https://github.com/user-attachments/assets/dcbdcb27-bbd2-49f3-911f-67ddc2799fe4)

*Example of the diffusion process (forward and reverse)*

## Key Features

- UNet architecture for noise prediction
- Customizable diffusion process parameters
- Training and sampling scripts
- Visualization utilities
- Support for both DDPM and DDIM sampling

## Mathematical Background

### Forward Process (q)

The forward process gradually adds Gaussian noise to the data according to a variance schedule `β₁,...,βₜ`:

`q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)`


### Reverse Process (p)

The reverse process learns to denoise by predicting the noise component:

`p_θ(xₜ₋₁|xₜ) = N(xₜ₋₁; μ_θ(xₜ,t), Σ_θ(xₜ,t))`


### Loss Function

The simplified objective (Ho et al., 2020):

`Lₜ₋₁ = E[||ε - ε_θ(xₜ,t)||²]`


where `ε` is the actual noise and `ε_θ` is the predicted noise.

## Dataset

The implementation uses the Sprites dataset (16x16 pixel images) from Hugging Face:

![Sprites Examples](https://github.com/user-attachments/assets/5d0f6364-49ce-4165-bc79-f4f013bef1d2)

*Example sprites from the dataset*

Dataset details:
- 1788 16x16 RGB images
- 5-dimensional context vectors
- Preprocessed with normalization (values in [-1, 1])

## Implementation Details

### Model Architecture

So what model do we use to do this magical 'denoising' step? We've looked a little at basic convolutional neural networks that take in an image and output something like a classification. And we've seen autoencoders that go from an image down to a latent representation and back to an output image. Perhaps one of these would be suitable?

![UNet Architecture](https://github.com/user-attachments/assets/f4466f06-28e4-4360-9e49-30de11aa4258)

One issue with a typical 'bottlekneck' architecture like an autoencoder is that by design they loose the details around exact pixel coordinates. To get around this, an architecture called the Unet was introduced. Originally designed for segmentation tasks, the architecture (shown above) passes information from high-resolution, early layers to later layers. These 'shortcuts' let the network use detailed features from the original image while also capturing more high-level semantic information from the deeper layers.

These networks turned out to be great at all sorts of image-to-image tasks. Colorization , segmentation and so on. These days, typical unet models incorporate ideas such as attention and can be built around pretrained 'backbones' like resnet-50 for transfer learning tasks.

The implementation below is a fairly typical modern Unet with one extra trick: a TimeEmbedding which encodes the time step (t) and lets the model use this as conditioning information by passing it in in the middle of the network. Take a peek at the code and see if you can figure out roughly what's going on in the forward pass


The UNet model consists of:
- Downsampling blocks
- Middle blocks with self-attention
- Upsampling blocks
- Residual connections throughout

```python
class Unet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):  # cfeat - context features
        super(Unet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  #assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)        # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2 #[10, 256, 4,  4]

         # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample
            nn.GroupNorm(8, 2 * n_feat), # normalize
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat), # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x)       #[10, 256, 8, 8]
        down2 = self.down2(down1)   #[10, 256, 4, 4]

        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)

        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)     # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        #print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")


        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out
```

## Training Process
### Hyperparameters:

* Timesteps: 1000

* β₁: 1e-4 to β₂: 0.02 (linear schedule)

* Batch size: 128

* Epochs: 40

* Learning rate: 1e-3

## Results
### Generated samples after training:
> DDPM Sampling
![DDPM Sampling](https://github.com/user-attachments/assets/42b5107e-406f-4d0c-8fcf-6734ebf00a2d)

> DDIM Sampling
![image](https://github.com/user-attachments/assets/37f2fdb6-ea58-4f2d-8524-a104ed2f5cd8)


## Dependencies
```python
pip install torch torchvision matplotlib numpy tqdm
```

## Download Dataset
```
!wget 'https://huggingface.co/datasets/ashis-palai/sprites_image_dataset/resolve/a24918819843abc0d1bee75a239024415081a87d/sprites_1788_16x16.npy'
```
```
!wget 'https://huggingface.co/datasets/ashis-palai/sprites_image_dataset/resolve/a24918819843abc0d1bee75a239024415081a87d/sprite_labels_nc_1788_16x16.npy'
```

## References
* Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.

* Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. ICLR.

* Hugging Face Sprites Dataset
