# Malaria Prediction Report

## Abstract
This project uses a feed-forward neural network to classify malaria positivity from 15 engineered features based on patient age and reported symptoms. Evaluated with the bundled `best_malaria.keras` model on the provided standardized test split, it achieved 99.17% accuracy and a ROC-AUC of 0.9997 (120 test records). These are results on a single, small dataset split and do not establish clinical suitability or performance on other populations.

## 1. Introduction
Malaria can present with fever, chills, headache, gastrointestinal symptoms, muscle pain, and fatigue. This project demonstrates a binary classifier that estimates whether a record is labeled malaria-positive from a set of symptom and age inputs. The output is an experimental model score, not a diagnosis.

## 2. Methodology

### 2.1 Dataset
The project includes 800 records in `malaria_dataset/malaria_raw.csv`, split into 560 training, 120 validation, and 120 test records. The prepared split files contain 15 input features and the `Malaria_Positive` target. The input columns include age, temperature, chills, headache severity, vomiting, diarrhea, muscle pain, fatigue, sweating, and nausea, along with engineered features. The prepared split values are standardized. The dataset's source and collection methodology are not documented in the project files.

### 2.2 Prediction Model
The notebook defines a fully connected neural network with 128-, 64-, and 32-unit hidden layers, batch normalization, ReLU activations, and dropout rates of 0.3, 0.3, and 0.2. The output is one sigmoid probability. Training uses Adam with a learning rate of 0.0005, binary cross-entropy, batches of 32, and up to 300 epochs. Early stopping and learning-rate reduction monitor validation AUC.

## 3. Implementation
The Streamlit app collects the ten base age and symptom inputs, derives five additional features, and passes the resulting 15 values to the saved Keras model. The app loads `best_malaria.keras` and the raw cohort from the project directory.

The original fitted preprocessing scaler is not included. The app reconstructs means and standard deviations from the full raw cohort, so its scores may differ from a pipeline using the original training-only scaler. For the test evaluation below, the model was run directly on the already-standardized test split.

## 4. Observations
On the 120-record test split, using a 0.5 classification threshold, the model achieved 99.17% accuracy, 100% precision and 98.00% recall for the malaria-positive class, and a ROC-AUC of 0.9997. The confusion matrix was 70 true negatives, 0 false positives, 1 false negative, and 49 true positives.

These metrics come from one small held-out split and should not be interpreted as evidence of clinical performance. The data source is undocumented, and no external validation is included. The notebook saves its checkpoint as `best_diabetes.keras`, while the app uses `best_malaria.keras`; the exact relationship between the notebook checkpoint and the bundled model is not documented. The notebook's example inference code also computes scaling statistics from the already-standardized training CSV, rather than saving and reusing the scaler fitted to the original training features. These artifact and preprocessing inconsistencies should be resolved before relying on the app's scores.

## 5. Conclusion
The project demonstrates an end-to-end malaria classification prototype, from tabular model training to a Streamlit symptom-entry interface. The bundled model performs strongly on the supplied test split, but the small dataset, undocumented data source, missing fitted scaler, and checkpoint naming mismatch limit reproducibility and generalization. The app is for educational demonstration only and is not a medical diagnosis tool.

## 6. GitHub Repository
- Project repository: [Generative-AI---2026](https://github.com/DelandMoore/Generative-AI---2026).

- Training notebook: [Deland_Malaria_prediction_ANN.ipynb](https://github.com/DelandMoore/Generative-AI---2026/blob/master/1.5_introduction_to_dl/Task_malaria_prediction/Deland_Malaria_prediction_ANN.ipynb).
- Trained model: [best_malaria.keras](https://github.com/DelandMoore/Generative-AI---2026/blob/master/1.5_introduction_to_dl/Task_malaria_prediction/malaria_saved_files/best_malaria.keras).
.
- Streamlit deployment URL: https://malaria-predictor5.streamlit.app/
