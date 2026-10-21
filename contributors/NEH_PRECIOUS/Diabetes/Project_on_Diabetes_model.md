# Diabetes Prediction with an Artificial Neural Network

## Abstract

This project builds a Keras ANN that predicts whether a patient has diabetes from clinical measurements in the Pima Indians Diabetes dataset. It covers EDA, missing-value handling, feature engineering, scaling, model training, and saving everything needed for inference.

## 1. Introduction

The dataset has 768 patient records, eight clinical features, and a binary Outcome label. About 35% of records are positive. The project focuses on careful data preparation and a regularized network, then saves the model and preprocessing objects so new patients are scored the same way as the training data.

## 2. Exploratory Analysis

- Glucose is the strongest predictor (correlation 0.50 with Outcome).
- Insulin (0.38), BMI (0.32), and SkinThickness (0.30) also correlate with Outcome.
- Diabetic patients tend to be older, with higher BMI and more pregnancies.

## 3. Data Preparation

- Missing values: Glucose, BloodPressure, SkinThickness, Insulin, and BMI are filled with training-set medians (117.0, 72.0, 28.0, 102.5, 32.05).
- Feature engineering: six features added, giving 14 inputs: Glucose_x_BMI, Age_x_BMI, impaired_glucose, diabetic_glucose, is_obese, older_patient.
- Splits: 537 train, 115 validation, 116 test, each with about 35% positives.
- Scaling: StandardScaler fitted on the training set only.

## 4. Model Architecture

- Input: 14 features
- Dense 128 → BatchNorm → ReLU → Dropout 0.3
- Dense 32 → BatchNorm → ReLU → Dropout 0.3
- Dense 32 → ReLU → Dropout 0.2
- Output: Dense 1, sigmoid

Training setup: Adam (learning rate 0.0005), binary cross-entropy, metrics: accuracy, AUC, precision, recall.

## 5. Project Files

- diabetes_ann.keras: trained model
- diabetes_eda.png: EDA figure
- diabetes_train.csv, diabetes_val.csv, diabetes_test.csv: engineered, unscaled splits
- diabetes_prepared.pkl: scaled arrays, scaler, feature list, medians
- diabetes_inference_kit.pkl: scaler parameters, feature list, medians
- diabetes_metadata.json: same inference information in JSON

## 6. Inference Workflow

New Record → Fill Missing Values → Engineer Features → Scale → ANN → Diabetes Probability

## 7. How to Create

1. Set up a Python project and virtual environment.
2. Install pandas, numpy, scikit-learn, matplotlib, and keras.
3. Load the dataset and run the EDA.
4. Fill missing values with training medians.
5. Create the engineered features.
6. Split into train, validation, and test sets.
7. Fit the scaler on the training set and transform all sets.
8. Build, compile, and train the ANN.
9. Evaluate on the test set.
10. Save the model, prepared data, inference kit, and metadata.

## 8. Problems Faced

- Handling missing values without leaking test information.
- Overfitting on a small dataset, addressed with dropout and batch normalization.
- Class imbalance, so precision, recall, and AUC are tracked alongside accuracy.
- Keeping preprocessing identical between training and inference.
- scikit-learn version warning when loading the saved scaler.

## 9. Conclusion

This project shows a complete tabular prediction workflow: data preparation, a regularized ANN, and a reusable inference kit. It can be extended with model comparison, threshold tuning, or a simple web interface.

This model is for learning purposes and is not a medical diagnostic tool.

## 10. GitHub Repository 

Project resipotary : https://github.com/negprecious69-commits/Generative_Ai/blob/main/1.5_introduction_to_dl/heart_disease_ann.keras
