# Score-Based Models for Toy 2D Mixture of Gaussians

This repository explores **Score-Based Generative Modeling** using a mixture of two Gaussians in 2D space. The project covers analytical score computation, training of score networks via score matching, and multiple sampling methods including **Deterministic Sampling**, **Langevin Dynamics**, and **Annealed Langevin Sampling**.

---

## 🧪 Toy Dataset

- **Distribution**: Mixture of 2 Gaussians (randomized with `student_number` seed)
- **Samples**: 5000 total (80% train / 20% test)
- **PDF Definition**:

Let `p(x) = w1 * N(x; mean1, cov1) + w2 * N(x; mean2, cov2)`

- **Visualizations**:
  - Heatmap of the PDF over `[-15, 15]^2`
  - Sample scatter plot with train/test split

---

## 📐 Analytical Score Function

The score function is defined as:

```text
∇ log p(x) = [w1 * N(x; mean1, cov1) * ∇ log N(x; mean1, cov1) +
              w2 * N(x; mean2, cov2) * ∇ log N(x; mean2, cov2)] / p(x)
```

Where:

```text
∇ log N(x; mean, cov) = -inv(cov) * (x - mean)
```

Implemented in `score_function_gt(x)` as a ground-truth baseline.

---

## 🤖 Score Network (Learned Score Function)

### Model: `ScoreNet`
- Input: (x1, x2, sigma)
- Architecture:
  - Linear(3 → 16) → ReLU
  - Linear(16 → 16) → ReLU
  - Linear(16 → 2)

### Loss Function: `score_matching_loss`
- Adds Gaussian noise `ε ~ N(0, σ²I)`
- True score: `s*(x) = ε / σ²`
- Predicted score: `ŝ(x) = f_theta(x + ε, σ)`
- Loss:

```text
L = 0.5 * E[ || ŝ(x) - (ε / σ²) ||² ]
```

### Training Strategy:
- Random `σ ∈ [1, 20]`
- Score network learns to approximate the gradient of log-density via MSE loss

---

## 🧭 Sampling Methods

### 1. Deterministic Sampling
```text
x_{t+1} = x_t + η * s(x_t)
```

### 2. Langevin Dynamics
```text
x_{t+1} = x_t + η * s(x_t) + sqrt(2 * η) * z_t,  z_t ~ N(0, I)
```

### 3. Annealed Langevin Dynamics
```text
x_{t+1} = x_t + 0.5 * η * s(x_t, σ_t) + sqrt(η) * z_t
```

Where `σ_t` is annealed linearly from `σ_start` to `σ_end`.

---

## 🎯 Final Experiments

- Trained models with `σ ∈ {1, 3, 7, [1, 20]}`
- Visualized vector fields learned by the network:
  - At various noise scales
- Compared generated samples:
  - Deterministic Sampling
  - Langevin Dynamics
  - Annealed Langevin Dynamics

---

## 📌 Summary

| Feature                 | Description                               |
|------------------------|-------------------------------------------|
| Data                   | Mixture of 2 Gaussians (2D)               |
| Score Function         | Analytic and learned                      |
| Loss                   | Noise-perturbed score matching            |
| Sampling Methods       | Deterministic, Langevin, Annealed         |
| Evaluation             | PDF heatmaps, quiver plots, trajectories  |

---

## 🛠️ How to Run

```bash
# Train the score model
train_score_model(model, train_data, sigma_range=[1, 20], epochs=100)

# Sample using Langevin
langevin_samples = langevin_sampling(model, start_points, sigma, step_size, steps)

# Sample using Annealed Langevin
annealed_samples = annealed_langevin_sampling(model, start_point, sigma_start, sigma_end, step_size, steps)
```

---

This repository is designed as an **interactive tutorial** and **experimental tool** for learning the mechanics of score-based generative modeling. It provides both theoretical foundations and practical demonstrations in a 2D Gaussian setup.


