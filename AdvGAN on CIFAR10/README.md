
# AdvGAN: Adversarial Example Generation Using GANs

This project implements **AdvGAN**, a technique for generating **adversarial examples** using GANs. Adversarial examples are inputs modified just slightly — often imperceptibly to the human eye — to cause a trained model to misclassify them.

---

## What Are Adversarial Examples?

Adversarial examples exploit the sensitivity of deep networks to small perturbations. Given an input `x` and a trained model `f(x)`, the goal is to find a modified input `x'` such that:


*f(x') ≠ f(x)  and  ||x' - x|| is small*


---

## Motivation: Why AdvGAN?

Compared to classical attack methods like **FGSM** and **PGD**, AdvGAN offers:

| Method     | Strengths                            | Weaknesses                     |
|------------|---------------------------------------|--------------------------------|
| **FGSM**   | Fast and easy to compute              | Weaker, one-step               |
| **PGD**    | Stronger attacks                      | Computationally expensive      |
| **AdvGAN** | Realistic, reusable adversarial examples | Requires training the GAN   |

AdvGAN learns a **generative model** to produce adversarial perturbations efficiently at inference time.

---

## Model Architecture

AdvGAN consists of:
- **Generator (G)**: Learns a mapping from clean images to perturbations.
- **Discriminator (D)**: Encourages generated samples to be indistinguishable from real inputs.
- **Target Model (f)**: Pretrained victim classifier.

The adversarial sample is generated as:


*x' = x + G(x)*


To ensure imperceptibility, the generator is trained with multiple losses.

---

## Loss Functions

### 1. **Adversarial Loss (for fooling the classifier)**

For untargeted attack:


*L_adv = CrossEntropy(f(x + G(x)), y_true)*

For targeted attack:


*L_adv = CrossEntropy(f(x + G(x)), y_target)*

---

### 2. **Reconstruction Loss (L2 Norm)**

Ensures the adversarial image remains visually close to the original:


*L_rec = ||G(x)||²*

---

### 3. **Discriminator Loss**

Standard GAN loss (Binary Cross Entropy), encouraging realism:


*L_GAN = E_x[log D(x)] + E_x[log(1 - D(x + G(x)))]*

---

### 🔗 Total Loss

The final objective is a weighted combination:


*L_total = λ1 * L_adv + λ2 * L_rec + λ3 * L_GAN*

---

## Attack Modes

### Untargeted Attack
- Aim: Any incorrect prediction.
- Effect: Drop in accuracy.

> *Untargeted Attack Samples*  
![Untargeted](https://github.com/user-attachments/assets/10724a81-543b-42bf-9412-1d7a7667504f)

---

### Targeted Attack
- Aim: Force model to predict a specific (wrong) label.
- Harder to achieve but more precise.

> *Targeted Attack Samples*  
![Targeted](https://github.com/user-attachments/assets/678e6a39-9b42-4ac4-a2f9-e1739abb7946)

---

### Class Prediction Shift
- Polar chart shows prediction distribution before and after attack.

> *Untargeted Polarity Plot*  
![Polarity](https://github.com/user-attachments/assets/8a79353e-2938-44f4-aebd-f1a0acceb225)

> *Targeted Polarity Plot*  
![Polarity](https://github.com/user-attachments/assets/5eb1554a-21fc-437a-852a-5e6399e788a1)

---

### Training Curve
- Demonstrates generator/discriminator convergence.

> *Loss Over Time*  
![Loss Curve](https://github.com/user-attachments/assets/738535f1-54d5-49a0-b7ea-818ad6ad4e22)

---

## Evaluation

| Metric                     | Value   |
|----------------------------|---------|
| Accuracy (clean test set)  | 81.13%  |
| Attack success (untargeted)| 63.14%  |
| Attack success (targeted)  | 53.83%  |

> Class-wise breakdown (untargeted):

| Class | Accuracy | Attack Success Rate |
|-------|----------|---------------------|
| 0     | 84.10%   | 44.40%              |
| 1     | 82.00%   | 74.30%              |
| 2     | 74.20%   | 48.60%              |
| ...   | ...      | ...                 |
| 9     | 82.50%   | 78.20%              |

---

## White-box vs Black-box Attacks

- **White-box**: AdvGAN uses the gradients of the target model directly during training.
- **Black-box**: A surrogate model is trained to mimic the target, then attacked similarly.

---

## Key Insights

- GANs offer **data-driven adversarial generation** that can generalize across inputs.
- Even high-confidence models can be **fooled silently** with imperceptible changes.
- Visual inspection confirms perturbations are minimal yet effective.

---

## References

- [AdvGAN (Xiao et al., 2018)](https://arxiv.org/abs/1801.02610)
- FGSM: Goodfellow et al., 2015
- PGD: Madry et al., 2017

---
