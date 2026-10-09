# Heart Disease Prediction Using Artificial Neural Networks

## 1. Project Overview

This project develops a machine-learning model for predicting whether a patient is likely to have heart disease based on a collection of medical, physical, and lifestyle-related features.
The project uses an **Artificial Neural Network (ANN)** implemented with **TensorFlow/Keras**. The model receives 20 patient features as numerical input and produces a probability between `0` and `1`.
The probability represents the model's estimated likelihood that the patient belongs to the **heart disease** class.
The project includes:

* A trained neural network
* Training, validation, and testing datasets
* Feature preprocessing information
* A saved Keras model
* A preprocessing/metadata file
* Training-history visualization
* ROC curve
* Confusion matrix
* A Jupyter notebook containing the complete training pipeline

# 2. Project Objective
The main objective is to create a binary classification model capable of answering:
> **Does this patient have heart disease?**
The problem is treated as a **binary classification problem**: `0` and `1`
The neural network produces a sigmoid probability:
It is important to understand that the model's raw output is a **probability estimate**, while the `0.5` threshold converts that probability into a class prediction.

## 3 `Heart_Disese_training.ipynb`
This is the main Jupyter Notebook containing the model-development pipeline.
It contains the following stages:
1. Import libraries
2. Load datasets
3. Load preprocessing metadata
4. Build the neural network
5. Compile the model
6. Configure callbacks
7. Train the model
8. Plot training history
9. Evaluate the model
10. Generate predictions
11. Calculate ROC-AUC
12. Generate a confusion matrix
13. Save the model and preprocessing information
This notebook is essentially the **source code for the machine-learning experiment**.

# 5. Dataset
So the complete dataset contains **50,000 patient records**. The project separates the data into three datasets:
```text
Heart_Disease_train.csv
Heart_Disease_val.csv
Heart_Disease_test.csv
```
The dataset contains **20 input features** and one target column.The target column is:`Heart_Disease`

## Separate Dataset Splits
The project uses separate:

```text
Training set
Validation set
Testing set
```
# ANN Architecture
This project is a **binary heart disease classification system using an Artificial Neural Network**.
It takes 20 demographic, lifestyle, physical, and medical features as input.
The data is standardized using statistics calculated from the training data.
The neural network consists of:
```text
20 input features -> 128 neurons ->64 neurons ->32 neurons ->1 sigmoid output
```
Regularization techniques including:
```text L2 regularizationDropout
Batch Normalization
```are used to improve the model's ability to generalize.The model is trained using:
```text
Adam, Binary Cross-Entropy, 300 maximum epochs, Batch size 32,Learning rate 0.0005```

and evaluated using
- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

The final trained model is saved as:`Heart_Disease.keras` while the preprocessing information is saved in:`Heart_Disease_kit.pkl`The project therefore contains most of the components required to move from **machine-learning experimentation to model deployment**.

#streamlit interface
   A user interface was developed using Streamlit so that users could enter the relevant patient symptoms and characteristics through a web interface, after which the application would pass the information to the saved neural network. The model would return a probability, and a threshold such as 0.5 could then be used to classify the result as malaria or no malaria.

   `STREAMLIT URL`: https://heart-disease-prediction-model-5nczku288fvbhjreqbst9y.streamlit.app/
   
#GITHUB PROJECT LINKS

Link:https://github.com/NganyuNathan/Group-Projec
