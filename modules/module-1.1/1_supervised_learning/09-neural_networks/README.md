# 🧠 Lesson 09 — Neural Networks

> **Core Idea**: Neural networks are loosely inspired by the brain — layers of interconnected "neurons" that transform inputs through a sequence of non-linear functions. They can learn extraordinarily complex patterns from data, making them the engine behind modern AI.

---

## 📋 Table of Contents

1. [From a Single Neuron to a Network](#1-single-neuron)
2. [Activation Functions — Introducing Non-Linearity](#2-activation-functions)
3. [The Forward Pass — Making a Prediction](#3-forward-pass)
4. [The Loss Function](#4-loss-function)
5. [Backpropagation — How the Network Learns](#5-backpropagation)
6. [Gradient Descent Variants](#6-gradient-descent-variants)
7. [Regularisation for Neural Networks](#7-regularisation)
8. [Architecture Choices](#8-architecture)
9. [Python Implementation (Keras)](#9-python-implementation)
10. [Visual Summary](#10-visual-summary)

---

## 1. From a Single Neuron to a Network

A single artificial neuron takes multiple inputs, multiplies each by a weight, sums them up, adds a bias, and passes the result through an activation function:

```
                w₁
    input x₁ ──────┐
                w₂  │
    input x₂ ──────┤──► z = w₁x₁ + w₂x₂ + w₃x₃ + b ──► a = f(z) ──► output
                w₃  │
    input x₃ ──────┘
    bias b   ──────┘

z = the linear combination (pre-activation)
f = the activation function (introduces non-linearity)
a = the neuron's output (activation)
```

A **neural network** arranges these neurons in layers:

```
INPUT LAYER      HIDDEN LAYER 1    HIDDEN LAYER 2    OUTPUT LAYER
                   (neurons)          (neurons)
   x₁ ●──────────────● ────────────────● ────────── ŷ₁ (output)
   x₂ ●──────────────● ────────────────●
   x₃ ●──────────────● ────────────────●
                      ●                ●
The inputs are passed through successive transformations.
Each hidden layer learns increasingly abstract representations of the input.
```

A network with at least one hidden layer is a **universal function approximator** — given enough neurons, it can approximate any continuous function to arbitrary precision (Universal Approximation Theorem). This is the theoretical foundation of deep learning.

---

## 2. Activation Functions — Introducing Non-Linearity

Without an activation function, stacking linear layers produces... another linear function. Non-linear activations are what make neural networks powerful:

```
Sigmoid: σ(z) = 1/(1+e^{-z})          Output: (0, 1)
  Pros: Outputs probabilities. Smooth derivative.
  Cons: Vanishing gradient for very large/small z. Slow convergence.
  Use: Output layer for binary classification only.

Tanh: tanh(z) = (e^z - e^{-z})/(e^z + e^{-z})  Output: (-1, 1)
  Pros: Zero-centred (better gradient flow than sigmoid).
  Cons: Still suffers from vanishing gradients at extremes.
  Use: Occasionally in RNNs.

ReLU: f(z) = max(0, z)                Output: [0, ∞)
  Pros: Simple, fast, no vanishing gradient for positive z.
  Cons: "Dying ReLU" — neurons can get stuck outputting 0 forever.
  Use: DEFAULT CHOICE for hidden layers in most networks.

Leaky ReLU: f(z) = max(0.01z, z)     Output: (-∞, ∞)
  Pros: Fixes dying ReLU by allowing small negative gradients.
  Use: When dying ReLU is a problem.

Softmax: σ(z)ᵢ = e^{zᵢ} / Σe^{zⱼ}   Output: probability distribution
  Use: OUTPUT LAYER for multi-class classification only.

RULE OF THUMB:
  Hidden layers → ReLU (or Leaky ReLU, GELU for transformers)
  Output: binary classification → sigmoid
  Output: multi-class classification → softmax
  Output: regression → linear (no activation)
```

---

## 3. The Forward Pass — Making a Prediction

For a network with two hidden layers, the forward pass computes:

```
Layer 1: z¹ = X W¹ + b¹,    a¹ = ReLU(z¹)
Layer 2: z² = a¹ W² + b²,   a² = ReLU(z²)
Output:  z³ = a² W³ + b³,   ŷ = softmax(z³)  (for multi-class)

Where:
  X    = input matrix (batch_size × n_features)
  W^l  = weight matrix for layer l
  b^l  = bias vector for layer l
  a^l  = activations of layer l (the layer's output)
```

Each weight matrix contains the learnable parameters. A network is trained by adjusting these weights to minimise the loss function.

---

## 4. The Loss Function

The loss function measures how wrong the current predictions are:

```
Regression:              MSE = (1/m) Σ (yᵢ - ŷᵢ)²

Binary classification:   Binary Cross-Entropy = −(1/m) Σ [yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]

Multi-class:             Categorical Cross-Entropy = −(1/m) Σₛ Σₖ yₛₖ log(ŷₛₖ)
                         (sum over all samples s and classes k)
```

The total loss is averaged over a mini-batch of examples, then gradients are computed and weights updated.

---

## 5. Backpropagation — How the Network Learns

Backpropagation is the algorithm that computes the gradient of the loss with respect to every weight in the network. It uses the **chain rule** of calculus, propagating error signals backwards from the output to the input:

```
Forward pass:  X → Layer 1 → Layer 2 → Output → Loss L

Backprop:      ∂L/∂W³ = ∂L/∂ŷ × ∂ŷ/∂z³ × ∂z³/∂W³      ← output layer gradient
               ∂L/∂W² = ∂L/∂a² × ∂a²/∂z² × ∂z²/∂W²     ← hidden layer 2 gradient
               ∂L/∂W¹ = ∂L/∂a¹ × ∂a¹/∂z¹ × ∂z¹/∂W¹     ← hidden layer 1 gradient

Each arrow is an application of the chain rule.
The process is called "backprop" because gradients flow backwards through the network.

Update: W^l ← W^l − α × ∂L/∂W^l   (gradient descent step for each layer)
```

Modern deep learning frameworks (PyTorch, TensorFlow/Keras) handle backprop automatically via **automatic differentiation** — you just define the forward pass and the framework computes all gradients for you.

---

## 6. Gradient Descent Variants

```
Batch Gradient Descent:    Compute gradient on ENTIRE dataset before updating.
                            Very slow for large data. Stable convergence.

Stochastic GD (SGD):       Update after EACH example.
                            Fast but very noisy — loss bounces around.

Mini-batch GD:             Update after each BATCH of (typically 32-256) examples.
  ← THE STANDARD            Best of both worlds: fast + stable enough.

Popular optimisers:
  SGD + Momentum:  Remember previous gradient direction, smooths out noise.
  RMSprop:         Adaptive learning rate per parameter (divide by moving avg of gradient²).
  Adam:            Combines momentum AND RMSprop. Currently the most popular default.
    θ ← θ − α × m̂ / (√v̂ + ε)
    where m̂ = bias-corrected momentum, v̂ = bias-corrected second moment

  Adam defaults: lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8
```

---

## 7. Regularisation for Neural Networks

Neural networks can have millions of parameters — overfitting is a serious concern. Key regularisation techniques:

```
Dropout: Randomly "drop" (set to zero) a fraction of neurons during each training step.
  Layer(Dropout(rate=0.5)) → 50% of neurons disabled per training step.
  At test time: all neurons active, but outputs scaled by (1-rate).
  Effect: Prevents neurons from co-adapting → forces redundant representations.
  Use on dense layers, not on output layer.

L2 Regularisation (Weight Decay):
  Add λΣw² to the loss → penalises large weights.
  kernel_regularizer=regularizers.l2(0.001)

Batch Normalisation: Normalise the inputs to each layer.
  Stabilises training, allows larger learning rates.
  Reduces dependence on careful weight initialisation.
  Add between the linear transformation and activation function.

Early Stopping: Monitor validation loss. Stop training when it starts increasing.
  Prevents overfitting without explicit penalty terms.
```

---

## 8. Architecture Choices

```
Number of hidden layers:
  0 → linear model (same as logistic/linear regression)
  1 → can approximate most functions (but may need many neurons)
  2-5 → good for most tabular data problems
  10+ → "deep" network; needed for images, audio, text

Number of neurons per layer:
  Start with 64 or 128. Use the same for all hidden layers as a baseline.
  Decrease going deeper: e.g., 256 → 128 → 64.
  Too few: underfitting. Too many: overfitting + slow training.

Rule of thumb for tabular data: 2-4 layers, 64-512 neurons per layer.
For images: use CNNs. For sequences: use RNNs, LSTMs, or Transformers.
```

---

## 9. Python Implementation (Keras)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ─── Data ────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─── Build the model ─────────────────────────────────────────────────────
model = keras.Sequential([
    layers.Input(shape=(30,)),                  # 30 features
    layers.Dense(128, activation='relu',        # hidden layer 1
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),                 # stabilise training
    layers.Dropout(0.3),                        # prevent overfitting
    layers.Dense(64, activation='relu',         # hidden layer 2
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')       # binary output
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()  # prints total parameters

# ─── Train ───────────────────────────────────────────────────────────────
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)
history = model.fit(
    X_train_s, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# ─── Plot learning curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='Train loss')
axes[0].plot(history.history['val_loss'], label='Val loss')
axes[0].set_title('Loss Curve'); axes[0].legend()
axes[1].plot(history.history['accuracy'], label='Train accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val accuracy')
axes[1].set_title('Accuracy Curve'); axes[1].legend()
plt.show()

# ─── Evaluate ────────────────────────────────────────────────────────────
y_prob = model.predict(X_test_s).flatten()
y_pred = (y_prob >= 0.5).astype(int)
print(f"Test accuracy: {(y_pred == y_test).mean():.4f}")
```

---

## 10. Visual Summary

```
╔══════════════════════════════════════════════════════════════════╗
║                  NEURAL NETWORKS — OVERVIEW                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  INPUT → [Dense+ReLU] → [Dropout] → [Dense+ReLU] → [Output]    ║
║                                                                  ║
║  FORWARD PASS: Input flows forward through layers → prediction  ║
║  LOSS:         Measure error (cross-entropy or MSE)             ║
║  BACKPROP:     Compute ∂Loss/∂W for every weight (chain rule)   ║
║  UPDATE:       W ← W − α × ∂Loss/∂W  (gradient descent)        ║
║  REPEAT:       Until validation loss stops improving            ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  KEY REGULARISATION: Dropout + L2 + BatchNorm + Early Stopping  ║
║  KEY OPTIMISER: Adam (lr=0.001 is a great starting point)       ║
║  KEY RULE: Always scale inputs; monitor both train and val loss  ║
╚══════════════════════════════════════════════════════════════════╝
```
