# Heart Disease Prediction Report

## Abstract
This project uses a neural network to estimate the presence of heart disease from 20 demographic, lifestyle, medical-history, and clinical features. On the saved test run, it achieved 92.6% accuracy and a ROC-AUC of 0.9828. These results are preliminary and do not establish clinical suitability.

## 1. Introduction
Heart disease risk is associated with multiple factors, including age, blood pressure, cholesterol, smoking, diabetes, and family history. This project combines these inputs in a binary classifier that predicts whether the dataset label indicates heart disease.

## 2. Methodology

### 2.1 Dataset
The prepared data contains 50,000 records split into 35,020 training, 7,480 validation, and 7,500 test examples. The 20 input features include age, gender, BMI, lifestyle factors, medical history, blood pressure, heart rate, fasting blood sugar, and total cholesterol. The target is `Heart_Disease` (0 or 1). Feature values in the CSV splits are standardized using training-set statistics.

### 2.2 Prediction Model
The training notebook builds a feed-forward neural network with 128, 64, and 32-unit hidden layers, batch normalization, dropout, and a sigmoid output. It trains with binary cross-entropy and reports accuracy, AUC, precision, and recall.

## 3. Implementation
The project stores the trained Keras model and preprocessing values separately. The Streamlit app collects patient inputs, applies the same scaling, predicts a risk probability, and uses SHAP to explain feature contributions.

## 4. Observations
The saved test results report 92.6% accuracy, approximately 92% precision and recall for the heart-disease class, and a ROC-AUC of 0.9828. However, the notebook configures early stopping and model checkpointing with `mode="max"` while monitoring `val_loss`; lower loss is better. The recorded run restores epoch-1 weights, so correct both callbacks to `mode="min"` and retrain before relying on these results. The project files also do not document the dataset's source or establish performance on external populations.

## 5. Conclusion
The project demonstrates an end-to-end heart-disease prediction prototype, from standardized tabular data to an explainable Streamlit interface. Its reported test performance is promising for experimentation, but the training callback setting should be corrected and the model independently validated before any clinical use. It is not a medical diagnosis tool.

## 6. GitHub Repository
- Project repository: https://github.com/DelandMoore/Heart-Disease-Predictor
- Streamlit app: https://heart-disease-predictor5.streamlit.app/
- Trained model: https://github.com/DelandMoore/Heart-Disease-Predictor/blob/master/Heart_Disease.keras
- Preprocessing values: https://github.com/DelandMoore/Heart-Disease-Predictor/blob/master/Heart_Disease_kit.pkl
- Dependencies: https://github.com/DelandMoore/Heart-Disease-Predictor/blob/master/requirements.txt

The training notebook, `Heart_Disese_training.ipynb`, is currently available in the local project folder but is not in this GitHub repository.
