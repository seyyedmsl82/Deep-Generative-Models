# VAE and TripletVAE for MRI Tumor Localization

This repository implements and compares two deep generative models for brain MRI reconstruction and anomaly localization: a basic Variational Autoencoder (VAE) and an enhanced TripletVAE with perceptual and structural loss components.

## 🎯 Objective

Reconstruct healthy MRI brain images and use reconstruction error to identify abnormal (tumor) regions in external scans. The pipeline simulates weak supervision using only healthy data during training and tests anomaly detection on BraTS tumor slices.

---

## 🧠 Model Architecture

### VAE

- **Input**: 1×256×256 grayscale brain slice

**Encoder**:
- 6 × Conv2d layers with increasing channels (1→512) + BatchNorm + ReLU
- Flatten → Linear (8192 → 1024) for `mean` and `log_var`

**Decoder**:
- Linear (1024 → 8192) → reshape to (512×4×4)
- 6 × ConvTranspose2d layers (512→1) with BatchNorm and ReLU
- Final activation: Tanh

---

### TripletVAE

The TripletVAE enhances the VAE using:
- **Triplet training**: Anchor, Positive, Negative
- **Skip connections** from encoder to decoder
- **Gated Cross Skip (GCS)** blocks for better feature fusion
- **Multi-level reconstruction losses**

**Encoder**:
- Similar to VAE + returns feature maps x1–x6 for skip connections

**Decoder**:
- Injects GCS between decoder and encoder features
- Concatenates encoder features at multiple stages

**Losses**:
- KL Divergence (Anchor + Positive)
- L1 Loss (Anchor, Positive, Negative) at coarse scale
- L1 + SSIM Loss (Negative 256x256 full-scale recon)
- Triplet Loss on latent embeddings with MarginRankingLoss

> Architecture Visualization
![TripletVAE Architecture](https://github.com/user-attachments/assets/0fba9d3c-8c69-488d-b7cd-398f76b0d00e)

---

## 📦 Dataset

- **Healthy**: [IXI-T2 Slices](https://www.kaggle.com/datasets/haonanzhou1/ixit2-slices)
- **Tumor**: [BraTS 2020 - T2 and seg.nii](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

Samples from both datasets:

> IXI Healthy Samples
![IXI Healthy Samples](https://github.com/user-attachments/assets/087aded5-3cff-428e-a94a-e98d86a034a8)

> BraTS Patient Slices
![BraTS Patient Slices](https://github.com/user-attachments/assets/753d2d28-a55a-4703-8359-8ccc1bf2e2f5)

---

## ⚙️ Training Details

| Setting        | VAE              | TripletVAE        |
|----------------|------------------|-------------------|
| Batch Size     | 128              | 9                 |
| Epochs         | 50               | 50                |
| Optimizer      | Adam             | Adam              |
| Learning Rate  | 0.0005           | 0.0001            |
| Noise Injection| No               | 16×16 and 256×256 |
| Losses         | MSE + KL         | KL + L1 + SSIM + Triplet |

---

## 📈 Evaluation

### On IXI
- Train only on healthy slices
- Monitor loss and visualize reconstructions across epochs

> VAE Reconstruction
![Real Samples](https://github.com/user-attachments/assets/370f67bf-ac2c-4c72-ba98-54b6853bb4a7)
![VAE Reconstruction](https://github.com/user-attachments/assets/3bff320a-dc41-40d4-91a4-ce0f37240a19)



### On BraTS (Unseen Tumors)
- Inject test tumor slices
- Use `|X - X̂|` to identify anomalies
- Apply Dice score on segmentations and visualize over real mask

> BraTS Samples
![BraTS Samples](https://github.com/user-attachments/assets/88669300-3736-43b7-975a-8f385e9d2af4)

> VAE Reconstruction
![VAE Reconstruction](https://github.com/user-attachments/assets/a2409776-df00-41f0-a58b-86fe00d133a0)

> BraTS Samples
![BraTS Samples](https://github.com/user-attachments/assets/d77ca977-40de-4638-b63e-c47fc06e2b52)

> TripletVAE Output
![TripletVAE Output](https://github.com/user-attachments/assets/a4790684-48ba-4faa-8901-b96792a63ed4)

---

## ✅ Comparison Summary

| Feature                        | Simple VAE         | TripletVAE (Paper)   |
|-------------------------------|---------------------|-----------------------|
| Full 2D Output                | ❌                  | ✅                    |
| Skull-Stripping Preprocess    | ❌ (raw data)       | ✅                    |
| Anomaly Type                  | None                | Coarse + Full         |
| Skip Connections              | No                  | GCS (Cross Attention) |
| Triplet Loss                  | ❌                  | ✅                    |
| Number of Epochs              | 20–50               | 50+                   |

---

## 📌 Conclusion

- VAE performs well on in-domain data but fails to detect unseen anomalies.
- TripletVAE improves both reconstruction quality and localization of tumor regions.
- Structural (SSIM) and Triplet-based learning make it a better fit for unsupervised anomaly detection in medical imaging.

> Compare VAE vs TripletVAE on BraTS: better masks, sharper errors, and useful generalization.

