# Day 21 Learning Journal: Convolutional Neural Networks (CNNs) - Part 1

**Date:** 2026-05-13  
**Author:** Fonyuy-pounds  
**Module:** Module 04 - Advanced Neural Networks & Applications  
**Day:** 21/60

---

## Morning Lesson: Introduction to CNNs (45-60 min)

### What I Learned Today

**The Limitations of Dense Networks:**

- Fully connected layers treat every pixel as independent input (e.g., 28×28 image = 784 parameters before hidden layer)
- This ignores **spatial structure** — nearby pixels are correlated, far pixels usually aren't
- Dense networks scale poorly: 224×224×3 ImageNet images would need 150K+ input connections to first hidden layer
- No translation invariance: moving an object slightly changes all pixel values, requiring relearning the same pattern

**Convolutional Layers - The Solution:**

- **Core idea:** Use a small filter (e.g., 3×3) that scans across the image, computing dot products at each location
- **Parameter sharing:** The same filter weights are reused at every position. Drastically reduces parameters!
- **Local connectivity:** Each neuron connects only to a small receptive field, not the entire input
- **Stride:** How many pixels the filter moves each step (stride=1 is standard, stride=2 downsamples)
- **Padding:** Add zeros around image edges (same padding preserves size, valid padding shrinks)

**Mathematical Formulation:**

$$Y_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} W_{m,n} \cdot X_{i+m, j+n} + b$$

Where $W$ is the filter kernel, $X$ is the input, $(i,j)$ is the output position, and $k$ is filter size.

**Key Properties:**

- **Translation equivariance:** If input shifts, output shifts the same way
- **Locality:** Each output position depends only on nearby input (receptive field grows with depth)
- **Efficiency:** Number of parameters ∝ kernel size, not input size!
- **Hierarchical features:** Early layers learn edges/textures, deeper layers learn object parts and whole objects

**Pooling Layers:**

- **Max Pooling:** Take maximum value in each window (2×2 common). Reduces spatial dimensions, adds translation invariance
- **Average Pooling:** Take mean value. Smoother but less commonly used in modern networks
- **Benefits:** Reduces parameters, provides non-linearity, focuses on strong activations

**Architecture Components:**

- Input → Conv → ReLU → Conv → ReLU → MaxPool → Flatten → Dense → Softmax
- Modern architectures use stride-2 convolutions instead of pooling for downsampling
- Batch normalization after convolutions stabilizes training significantly

### What Confused Me

- [ ] How exactly does the receptive field grow with network depth?
  - Still unclear: If layer 1 has 3×3 kernel, and layer 2 has 3×3 kernel, what's the receptive field at layer 2 output?
  - **Note:** Need to draw this out or visualize to understand. Receptive field calculation: RF_l = RF_(l-1) + (kernel_size - 1) * stride_product

- [x] Why translation equivariance is a feature, not a bug
  - **Clarity gained:** Translation equivariance means the network "sees" the same pattern regardless of position. This is exactly what we want for vision! If a cat is in the bottom-right, the network should still recognize it as a cat.

- [x] The intuition behind "local connectivity" in CNNs
  - **Clarity gained:** Early in the network, neurons only need to see a small neighborhood to detect features like edges. As we go deeper, the effective receptive field grows, allowing neurons to combine edge detections into shape detections. This hierarchy is powerful and efficient.

- [ ] Parameter counting in CNNs with different layer configurations
  - Partially unclear: I can calculate it, but quickly in my head I mix up kernel size, number of filters, and stride effects.

### What I Want to Explore Next

- Implement convolution from scratch (nested loops, then vectorized NumPy)
- Visualize learned filters in layer 1 (what edge patterns emerge?)
- Test on MNIST (should be much faster than dense networks, fewer parameters)
- Understand dilated convolutions and grouped convolutions
- Study classic architectures: LeNet, AlexNet, VGG, ResNet

---

## Midday Exercise (30-45 min)

### Exercise 1: Conv Layer Mechanics

**Task:** Implement a single 2D convolution operation from scratch

- [x] Completed

**Findings:**

```text
Input shape: (28, 28)
Filter shape: (3, 3)
Output shape (valid): (26, 26)
Output shape (same padding): (28, 28)

Conv operation on single filter:
- Input region: (3×3 window)
- Filter: (3×3 learnable weights)
- Output: scalar (dot product + bias)
```

**Key learning:** Padding and stride directly determine output dimensions:
$$\text{output\_size} = \frac{\text{input\_size} - \text{kernel\_size} + 2 \times \text{padding}}{\text{stride}} + 1$$

### Exercise 2: Multiple Filters

**Task:** Implement convolution with multiple filters (creates feature maps)

- [x] Completed

**Results:**

- 1 filter on 28×28 image: produces 1 feature map (26×26)
- 32 filters on 28×28 image: produces 32 feature maps (26×26 each)
- Interpretation: Each filter learns a different feature detector (oriented edges, blobs, corners, etc.)
- Output shape: (26, 26, 32) — 91,904 activations per image!

**Parameter efficiency check:**

- Dense layer (784 inputs → 32 outputs): 25,088 parameters
- Conv layer (3×3, 1→32 filters): 320 parameters ⭐ (78x fewer!)

### Exercise 3: Pooling Operation

**Task:** Implement 2×2 max pooling

- [x] Completed

**Findings:**

```text
Before pooling: (26, 26, 32)
After 2×2 max pooling, stride 2: (13, 13, 32)
Spatial dimensions halved, channels unchanged
```

**Effect analysis:**

- Reduces computation in next layer by 4x (13×13 vs 26×26)
- Provides translation invariance: small shifts in input don't change max value in pooling window
- Removes fine-grained spatial information, preserves strong activations

### Exercise 4: Simple CNN Architecture

**Task:** Stack layers together: Conv → ReLU → Conv → ReLU → MaxPool → Flatten → Dense

- [x] Completed

**Architecture:**

```text
Input (28, 28, 1)
  ↓
Conv (32 filters, 3×3) → (26, 26, 32)
  ↓
ReLU → (26, 26, 32)
  ↓
Conv (64 filters, 3×3) → (24, 24, 64)
  ↓
ReLU → (24, 24, 64)
  ↓
MaxPool (2×2) → (12, 12, 64)
  ↓
Flatten → (9,216)
  ↓
Dense (128) → (128)
  ↓
ReLU → (128)
  ↓
Dense (10) → (10)
  ↓
Softmax
```

**Parameters count:**

- Conv1: 32 × (1×3×3 + 1) = 320 ✓
- Conv2: 64 × (32×3×3 + 1) = 18,496 ✓
- Dense1: 9,216 × 128 = 1,179,648
- Dense2: 128 × 10 = 1,280
- **Total: ~1.2M parameters** (only 10% are in convolutions!)

**Comparison to fully-connected:**

- Dense network (784 → 256 → 128 → 10): 201,474 parameters
- CNN above: 1,199,744 parameters
- **Trade-off:** CNN has more params but learns hierarchical features, generalizes better on vision tasks

---

## Evening Journal (15 min)

### Summary (3 Sentences)

1. **What I learned:** Convolutional neural networks exploit spatial structure in images through parameter sharing and local connectivity. By using small filters instead of full connectivity, CNNs dramatically reduce parameters while learning hierarchical representations (edges → shapes → objects).

2. **What confused me:** I'm still building intuition about receptive fields and how deep networks see increasingly large regions of the input. I need to visualize this more carefully, but I now understand why early layers detect local patterns while deeper layers learn global structure.

3. **What I want to explore:** I'm excited to implement backpropagation through convolutions (understanding how gradients flow through conv operations is non-trivial). I also want to visualize what filters learn and compare CNN performance on MNIST vs. the dense networks from earlier weeks.

---

## Resources Used

- [x] Lesson: L21.1_convolutional-networks
- [x] Lesson: L21.2_pooling-architectures
- [x] Goodfellow et al. (2016) - Chapter 9 (Convolutional Networks)
- [x] Exercises in `day21_exercise.py`
- [x] Visualization tools for filter outputs

---

## Code Review Checklist

- [x] 2D convolution correctly implements the sliding window with dot product
- [x] Multiple filters create multiple output channels
- [x] Padding modes (valid and same) handled correctly
- [x] Max pooling selects correct maximum values per window
- [x] Output shapes computed correctly for each layer
- [x] Parameter counting verified by hand
- [x] Architecture assembled correctly (Conv → ReLU → MaxPool → Dense)
- [x] Test forward pass completes without errors

---

## Practical Insights for Day 22-24

**For upcoming work on vision tasks:**

1. CNNs are much better than dense networks for image classification — use them!
2. Start with simple architecture (2-3 conv layers) before jumping to ResNet
3. Parameter efficiency matters: avoid massive filter counts early
4. Pooling provides regularization effect (translation invariance), helps prevent overfitting
5. Visualizing learned filters (layer 1 is interpretable!) builds intuition about what network learns
6. For the Module 04 capstone, consider using CNN if working with image data; dense networks if working with features from Banso text embeddings

---

## Connected Learning

**Links to previous days (Module 03):**

- **Day 19:** (Backprop II) — CNN backprop is more complex; need to reverse convolution operation
- **Day 17:** (Regularization) — Pooling provides implicit regularization through translation invariance
- **Day 16:** (Loss Functions) — Still use cross-entropy loss, but now with CNN feature extractor

**Links to future days (Module 04):**

- **Day 22:** Implement backpropagation through convolutions
- **Day 23:** Study classic architectures (LeNet, VGG, ResNet patterns)
- **Day 24:** Fine-tune pretrained models from Hugging Face
