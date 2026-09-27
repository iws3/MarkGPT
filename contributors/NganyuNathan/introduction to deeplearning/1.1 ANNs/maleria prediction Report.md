
# MALERIA PREDICTION REPORT

## AIM of the project
   this project focuses on developing an Artificial Neural Network (ANN) for malaria classification. The aim was to train a neural network that could use patient-related symptoms and other features to determine whether a patient is likely to be positive or negative for malaria. The project was implemented in Python using TensorFlow/Keras for the neural network and Scikit-learn for model evaluation. The trained model was also saved so that it could later be used in a prediction application.

## data anlysis and cleaning
   The dataset used in the project was gotten from Kaggle and divided into training, validation, and testing sets. The training set contained 560 records, while both the validation and test sets contained 120 
   The first stage of the project involved preparing the data for the neural network. The target variable was named `Malaria_Positive`, while the remaining columns were used as input features. The training, validation, and testing datasets were loaded separately, allowing the model to be trained only on the training set. 

## ANN actichecture and structure
   The model developed for the project was a feed-forward Artificial Neural Network. It consisted of an input layer followed by three main dense layers containing 128, 64, and 32 neurons respectively. ReLU activation was used in the hidden layers, while the final output layer contained one neuron with a sigmoid activation function. Batch normalization was applied after the first two dense layers, while dropout was used to remove all active neurons and  L2 regularization was used to penalize the weights. These techniques were included to reduce the possibility of overfitting and make the model more stable during training.

   For optimization, the Adam optimizer was selected to adjust the weights, Binary cross-entropy was used as the loss function because the task involved two possible classes. Accuracy, AUC, precision, and recall were also included as evaluation metrics. Several callbacks were included, such as EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint. These callbacks were intended to help control the training process, adjust the learning rate when necessary, and save the best version of the model

## model evaluation and result
   After training, the model was evaluated using the separate test dataset. And values of the test loss and  metric value were recorded. The ROC analysis produced a ROC-AUC score which was recorded. This indicates that the model was able to distinguish between the two classes effectively on the test data. The project also generated a confusion matrix to provide a more detailed view of the correct and incorrect classifications.

## saving the model
   The final neural network was saved as `malaria_ann.keras`, while an inference kit containing the feature information was saved as `malaria_inference_kit.pkl`. This makes it possible to load the trained model later and use it to make predictions on new patient information without retraining the entire network.

## streamlit interface
   A user interface was developed using Streamlit so that users could enter the relevant patient symptoms and characteristics through a web interface, after which the application would pass the information to the saved neural network. The model would return a probability, and a threshold such as 0.5 could then be used to classify the result as malaria or no malaria.
   **STREAMLIT URL:**

## Github links to the project
- maleria prediction github url: https://github.com/NganyuNathan/GenAI-/tree/master/0.4_introduction_to_dl/malerai%20training%20assignment
