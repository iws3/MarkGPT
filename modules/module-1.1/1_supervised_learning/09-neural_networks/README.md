# Neural Networks for Supervised Learning

## Fundamentals

Neural Networks are powerful deep learning models inspired by biological neurons that can learn complex non-linear transformations through multiple layers. In supervised learning, neural networks can be trained with backpropagation to optimize weights and biases, making them highly flexible for both classification and regression tasks. Modern neural networks with multiple hidden layers (deep learning) have revolutionized fields like computer vision, natural language processing, and speech recognition. Understanding neural network fundamentals is crucial for modern machine learning practitioners, although their black-box nature can make interpretability challenging.

## Key Concepts

- **Neurons and Layers**: Architecture design
- **Activation Functions**: ReLU, Sigmoid, Tanh
- **Backpropagation**: Gradient computation and weight updates
- **Learning Rate and Optimization**: SGD, Adam, RMSprop

## Applications

- Image classification
- Text classification and NLP
- Time series prediction
- Speech recognition
- Recommendation systems

---

[Go to Exercises](exercises.md) | [Answer the Question](question.md)



### Neural Network Fundamentals

Neural networks are computational models loosely inspired by biological neural networks, consisting of interconnected layers of artificial neurons. Each neuron computes a weighted sum of inputs plus a bias term, then applies a non-linear activation function. The network learns by adjusting weights and biases through backpropagation, which computes gradients of the loss function with respect to each parameter. Neural networks can approximate any continuous function to arbitrary precision (universal approximation theorem), making them extremely flexible. The depth (number of layers) and width (neurons per layer) are hyperparameters controlling model capacity. Deep networks can learn hierarchical representations where early layers detect simple patterns, middle layers combine them into complex patterns, and final layers make predictions.

### Activation Functions and Network Architecture

Activation functions introduce non-linearity, without which stacking layers would be equivalent to a single linear layer. Common choices include sigmoid σ(z) = 1/(1+e^-z), tanh, and ReLU f(z) = max(0, z). ReLU is preferred in modern networks due to training speed and simplicity. The network architecture—number of layers, neurons per layer, and connections—determines parameters and computational cost. Fully connected networks have each neuron connected to all previous layer neurons. Convolutional networks use weight sharing and local connections for images. Recurrent networks use temporal connections for sequences. Choosing appropriate architecture for the problem dramatically impacts performance and efficiency.

### Training Challenges and Advanced Techniques

Training deep networks presents challenges: vanishing/exploding gradients (gradients become very small/large during backpropagation), overfitting (learning noise instead of signal), and slow convergence. Batch normalization normalizes layer inputs, stabilizing training and allowing higher learning rates. Dropout randomly deactivates neurons during training, providing regularization. Residual connections (skip connections) enable training of extremely deep networks by alleviating gradient flow problems. Adam optimizer adapts learning rates per-parameter, often outperforming standard SGD. Weight initialization affects training speed; recent techniques like He initialization consider activation functions when initializing. Modern deep learning prioritizes debugging networks, understanding what they learn, and using regularization to prevent overfitting.