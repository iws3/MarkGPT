# Heart Disease Prediction App

## Description

This project is a machine learning application that predicts whether a person is likely to have heart disease based on health-related information.

The prediction model is an Artificial Neural Network (ANN), and the model is connected to a **Streamlit** application so users can enter their information and receive a prediction through a web interface.

## Technologies Used

* **Python** – Main programming language
* **TensorFlow/Keras** – Used to build and train the ANN model
* **Pandas & NumPy** – Used for handling and processing data
* **Scikit-learn** – Used for data preprocessing and model evaluation
* **Matplotlib** – Used for visualizing the model's performance
* **Jupyter Notebook** – Used for developing and testing the model
* **Streamlit** – Used to create the web application

## How I Built It

The project was built in several stages:

1. **Data Preparation**
   The heart disease dataset was prepared and the required features were selected.

2. **Data Preprocessing**
   The data was cleaned and prepared for training. Numerical features were scaled so that they could be properly used by the neural network.

3. **Model Development**
   An Artificial Neural Network was created using TensorFlow/Keras. The model was trained to classify the input into two possible outcomes: heart disease or no heart disease.

4. **Model Evaluation**
   The trained model was evaluated using metrics such as accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC.

5. **Model Saving**
   After training, the model was saved so that it could be used later without training it again.

6. **Streamlit Application**
   A Streamlit interface was created where users can enter the required health information. The saved model and preprocessing information are then used to make a prediction.

## How It Works

The user enters health information such as age, gender, BMI, blood pressure, cholesterol, blood sugar and other relevant factors.

The application takes these values, prepares them in the same way as the training data, and sends them to the trained ANN model. The model then analyzes the information and produces a prediction.

The result is displayed on the Streamlit application as either **Heart Disease** or **No Heart Disease** and the probability in which a person can have heart disease

## Main Features

The application allows users to:

* Enter a patient's name
* Enter health-related information
* Submit the information for prediction
* Receive a heart disease prediction through the web interface

## Challenges Faced

Some of the challenges faced during the project included:

* Finding the right dataset or a suitable dataset
* Choosing a suitable ANN architecture and number of neurons.
* Dealing with compatibility issues between TensorFlow, Keras and other Python packages.

## Conclusion

This project provided practical experience in building a machine learning model from data preparation to deployment. It helped demonstrate how an ANN can be trained for heart disease prediction and then integrated into a Streamlit application that allows users to interact with the model through a simple interface.

## Github Repositories
- Data preparation and cleaning: https://github.com/AkongnwiDarius/Generative_Ai/blob/master/1.5_introduction_to_dl/Heart%20Disease/Data_cleaning.ipynb
- Model Training: https://github.com/AkongnwiDarius/Generative_Ai/blob/master/1.5_introduction_to_dl/Heart%20Disease/Heart_Disease.ipynb
- Streamlit application: https://heart-disease-application000.streamlit.app/