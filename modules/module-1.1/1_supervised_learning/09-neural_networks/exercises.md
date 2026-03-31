# 📝 Neural Networks — Exercises

## Exercise 1 — Single Neuron from Scratch
Implement a single neuron in pure NumPy. It should accept input features, weights, and a bias. Implement three activation functions: sigmoid, tanh, and ReLU. Plot all three on the same graph for z in [-5, 5]. Then compute the output of your neuron for a simple input (e.g., [0.5, -0.3, 0.8]) with random weights. Compute the derivative of each activation function — this is what gets used during backpropagation.

## Exercise 2 — Backpropagation for a 2-Layer Network
Build a complete 2-layer neural network from scratch using only NumPy. Architecture: 2 inputs → 4 hidden neurons (ReLU) → 1 output (sigmoid). Implement forward pass, binary cross-entropy loss, and backpropagation. Train it to learn the XOR function: inputs [(0,0),(0,1),(1,0),(1,1)], outputs [0,1,1,0]. A single-layer network cannot learn XOR — but a 2-layer one can! Plot the loss over 10,000 training iterations.

## Exercise 3 — Learning Rate Experiment
Build a Keras dense network on the breast cancer dataset. Train it with learning rates: 0.0001, 0.001, 0.01, 0.1. Plot the validation loss curve for all four on the same graph. Identify: which learning rate causes divergence? Which converges too slowly? Which finds the sweet spot? Does Adam automatically adapt, or does the initial learning rate still matter?

## Exercise 4 — Dropout and Overfitting
On a small dataset (200 training samples of breast cancer, same test set), train: (1) a wide network (512 neurons per layer) without dropout, (2) the same network with Dropout(0.5) after each layer. Plot training and validation accuracy for both. The undropped network should clearly overfit — show this visually. Explain in a comment why dropout acts as regularisation.

## Exercise 5 — Batch Normalisation Effect
Add BatchNormalization layers to your network (before the activation function, after the Dense layer). Train a network without BatchNorm and one with BatchNorm, both on the breast cancer dataset. Compare: number of epochs to convergence (use early stopping), final validation accuracy, and the stability of the training loss curve (does it fluctuate a lot?). BatchNorm typically converges faster and is more stable.

## Exercise 6 — Architecture Search
Design a systematic experiment to find the best architecture for the breast cancer dataset. Try: 1, 2, and 3 hidden layers, each with 32, 64, 128, and 256 neurons. That's 4×3 = 12 architectures. Use 5-fold cross-validation for each (or use a validation split for speed). Create a table of validation accuracy for all 12 architectures. Which architecture wins? Does adding more layers always help?

## Exercise 7 — Optimiser Comparison
Train the same network architecture on the breast cancer dataset using: SGD (no momentum), SGD with momentum=0.9, RMSprop, and Adam. Plot all four validation loss curves on the same graph. Which converges fastest? Which is most stable? Try SGD with a learning rate scheduler (reduce by half every 20 epochs). Does that help SGD compete with Adam?

## Exercise 8 — Neural Network for Regression
Use the California Housing dataset. Build a dense network for regression (output layer = 1 neuron, no activation = linear). Use MSE as the loss function. Train with Adam. Compare RMSE and R² against LinearRegression, RandomForestRegressor, and GradientBoostingRegressor. Plot learning curves. For tabular regression, gradient boosting often outperforms neural networks — verify whether that's true for this dataset.

## Exercise 9 — Weight Initialisation Matters
Train two identical networks on the same data but with different weight initialisations: (1) all weights initialised to zero, (2) glorot_uniform (the default). Plot the training loss for both. With zero initialisation, all neurons in a layer are identical and learn the same thing — this is the "symmetry breaking" problem. The zero-init network should fail to learn properly. Explain why in a comment.

## Exercise 10 — MNIST Digit Classification
Load the MNIST handwritten digits dataset (`tensorflow.keras.datasets.mnist`). Build a dense neural network (no convolutions — just flatten the 28×28 image into 784 inputs). Design your own architecture and train it. Report test accuracy (the best dense networks achieve about 98%). Visualise 5 misclassified examples. Then add BatchNormalization and Dropout and retrain. Does performance improve? Finally, compare your best dense network against a simple ConvNet — the gap shows why CNNs were invented.
