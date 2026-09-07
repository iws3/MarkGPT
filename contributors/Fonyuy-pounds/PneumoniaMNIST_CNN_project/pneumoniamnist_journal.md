# Project Journal — PneumoniaMNIST Lightweight CNN

**Author:** Fonyuy Patrick  
**Repository:** [cnn-pneumonia-medmnist](https://github.com/Fonyuy-pounds/cnn-pneumonia-medmnist.git)  
**Date:** August 2026

---

## Overview

This project focuses on building a lightweight Convolutional Neural Network (CNN) to classify chest X-ray images as either Normal or Pneumonia. The dataset used is PneumoniaMNIST, part of the MedMNIST collection, which provides 28×28 grayscale medical images. The primary goal is not just accuracy but high recall — ensuring that as few pneumonia cases as possible are missed, since a false negative in medical diagnosis can be life-threatening.

---

## Dataset Description

The PneumoniaMNIST dataset contains 5,856 chest X-ray images split into training, validation, and test sets.

| Split | Samples |
| ------- | --------- |
| Training | 4,708 |
| Validation | 524 |
| Test | 624 |

Each image is 28×28 pixels with a single grayscale channel. The dataset is imbalanced — approximately 74% of training samples are pneumonia cases and 26% are normal. This imbalance is important because accuracy alone can be misleading; a model that always predicts pneumonia would achieve 74% accuracy but would be clinically useless.

---

## Model Architecture

The CNN architecture is deliberately simple and lightweight. It consists of two convolutional blocks followed by a classification head.

The first convolutional layer uses 16 filters of size 3×3 to detect basic features such as edges and textures. This is followed by a max pooling layer that reduces spatial dimensions from 26×26 to 13×13. The second convolutional layer uses 32 filters to learn more complex patterns like white patches and lung boundaries, followed by another pooling layer that reduces dimensions to 5×5. The flattened output of 800 values is passed to a single dense neuron with sigmoid activation, which outputs a probability between 0 and 1.

The total number of trainable parameters is only 5,601. This lightweight design is intentional — it prevents overfitting on a relatively small dataset, trains quickly, and demonstrates that complex architectures are unnecessary for simple binary classification on small images.

---

## Training Process

The model was trained using the Adam optimizer with a learning rate of 0.001 and binary cross-entropy as the loss function. Training was set for up to 35 epochs with a batch size of 32.

Two callbacks were used during training. EarlyStopping monitored validation loss with a patience of 5 epochs, automatically stopping training when improvement stalled and restoring the best weights. ReduceLROnPlateau reduced the learning rate by half when validation loss plateaued for 3 consecutive epochs, allowing the model to fine-tune its weights.

Training converged well. The best epoch was epoch 31, after which the learning rate was reduced twice to squeeze out final improvements. Training and validation curves remained close, indicating minimal overfitting.

---

## Results

The model was evaluated on the hold-out test set of 624 images that were never seen during training.

| Metric | Value |
| -------- | ------- |
| Accuracy | 85.26% |
| Precision | 81.84% |
| Recall | 98.21% |
| F1-Score | 89.28% |

The confusion matrix reveals the clinical significance of these numbers. Out of 390 actual pneumonia cases, the model correctly identified 383 and missed only 7 — a miss rate of just 1.8%. Out of 234 normal cases, 149 were correctly identified and 85 were falsely flagged as pneumonia — a false alarm rate of 36.3%.

This trade-off is deliberate. Missing pneumonia is dangerous because a patient sent home untreated could deteriorate. A false alarm is manageable because a physician can review the case and order additional tests. The model prioritizes patient safety over reducing false alarms.

---

## Key Design Decisions

**Recall over accuracy.** The model is optimized to catch pneumonia cases even at the cost of more false positives. This reflects the clinical reality that false negatives are more dangerous than false positives.

**No horizontal flipping.** Chest X-rays are anatomically asymmetric — the heart is on the left side. Flipping an X-ray horizontally creates an anatomically impossible image where the heart appears on the right. This would teach the model incorrect features and potentially harm performance.

**Separated data pipeline.** Data preparation and training are strictly separated. The data preparation script downloads the dataset, normalizes pixel values, and saves processed arrays as files. The training script loads these files without any preprocessing logic. This ensures reproducibility, modularity, and easier debugging.

**Lightweight architecture.** With only 5,601 parameters, the model is efficient and practical. It demonstrates that small, well-designed networks can perform excellently on small images without the need for deep architectures.

---

## Repository Structure

The repository is organized into clear, modular scripts. The data preparation script handles all preprocessing. The model script defines the CNN architecture. The training script loads prepared data and trains the model. The evaluation script calculates metrics and generates visualizations. Generated data files, saved models, and plots are excluded from version control.

---

## Lessons Learned

This project reinforced several important concepts. First, accuracy is not always the right metric — for imbalanced medical data, recall and precision provide a clearer picture of real-world performance. Second, medical imaging requires careful thought about augmentation — not all standard image transformations are appropriate. Third, clean separation of data preparation from training makes a project more reproducible and easier to debug. Finally, a lightweight model can achieve strong results when well-designed, without unnecessary complexity.

---

## Tools Used

TensorFlow and Keras for model development, NumPy for array operations, Matplotlib and Seaborn for visualization, scikit-learn for evaluation metrics, and MedMNIST for dataset access.
