# RealNVP for FashionMNIST Density Estimation and Generation

This repository implements a **Real-valued Non-Volume Preserving (RealNVP)** flow-based generative model using PyTorch. The model is trained on the **FashionMNIST** dataset to learn a bijective mapping from data space to a latent space, enabling efficient sampling and exact likelihood estimation.

---

## 🧠 Model Architecture

The RealNVP model is built with multiple **Affine Coupling Layers** and **Permutation Layers**, and optionally includes **Linear Batch Normalization**.

### 🔁 Coupling Layer
- Splits the input into two parts: x = [x_a, x_b]
- Transforms x_b using: y_b = x_b * exp(s(x_a)) + t(x_a)
- Log-determinant Jacobian: -sum s(x_a)

### 🔀 Permutation Layer
- Shuffles feature dimensions between coupling layers to enhance interaction across components.

### ⚙️ s / t Networks
- Each coupling layer has two MLPs:
  - `s_network`: Outputs scale vector s(x_a)
  - `t_network`: Outputs translation vector t(x_a)

### 🌐 Prior
- The base distribution is a standard multivariate Gaussian:
z ~ N(0, I)

> RealNVP Architecture
![RealNVP Architecture](https://github.com/user-attachments/assets/3405e937-56db-4901-beb7-dcd16b8d7e1e)

---

## 📦 Dataset

- **FashionMNIST**: 28x28 grayscale clothing images
- **Split**: 80% training, 20% validation, standard test set
- **Transform**: Normalized to [-1, 1] with `transforms.Normalize((0.5,), (0.5,))`

---

## ⚙️ Training Details

- **Input dimension**: 784 (28x28)
- **Coupling Layers**: 8 total (each with a permutation layer after)
- **Hidden units per s/t net**: 1024
- **Batch size**: 512 (train/val), 64 (test)
- **Optimizer**: Adam (lr=0.0001)
- **Epochs**: Customizable

> Training loss consists of:
- Negative Log Likelihood (from prior)
- Log-determinant of Jacobian
- Optional: Batch norm regularization loss

> Loss Plot
![Loss Plot](https://github.com/user-attachments/assets/90c0e9f7-570c-4924-b3d7-07dc476ab115)

---

## 🧪 Evaluation Metrics

- **Log-likelihood**: Exact density evaluation
- **Generated samples**: Forward from Gaussian latent z

> Sample Generated Output

![Sample Generated Output](https://github.com/user-attachments/assets/ad58cac9-0312-4c19-b855-dd99762fe2dc)

---

## ✅ Results

- Converging training/validation loss with 8 coupling layers
- Realistic FashionMNIST samples generated from base Gaussian
- High-quality density estimation with interpretable latent structure

---

## 🛠️ Usage

```bash
# Train
realnvp.train_realnvp(train_loader, val_loader, epochs=50, optimizer=optimizer, device=device)

# Test
realnvp.test_realnvp(test_loader, device)

# Sample
samples = realnvp.sample(batch_size=16)
```

---

## 📌 Summary

RealNVP offers:
- Exact likelihood computation
- Invertible mapping
- Efficient sampling

This makes it an elegant alternative to VAEs and GANs when interpretability and likelihood evaluation are crucial.

