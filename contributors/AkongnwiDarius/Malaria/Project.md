# Project
# Malaria Prediction with an Artificial Neural Network (ANN)
## Abstract

This project builds an artificial neural network (ANN) that predicts whether a patient has malaria from basic symptoms and personal details such as age, fever temperature, chills, headache severity and fatigue. The model was trained on 560 patient records, tuned on 120 validation records and tested on 120 records it had never seen. On the test set the saved results show an accuracy of about 99 percent and an ROC-AUC of 1.0, with one missed malaria case and no false alarms implied by the report. These notes explain how the data was organised, how the network was designed and trained, how it was evaluated, and what problems and limits were found.

## Introduction

Malaria is diagnosed with laboratory tests, but clinics do not always have quick access to them. A model that can estimate the likelihood of malaria from symptoms that are easy to record could help health workers decide who needs testing first. This project is a learning exercise that tests that idea on a patient dataset using deep learning.

The problem is a binary classification: for each patient the model gives a probability between 0 and 1, and a probability of 0.5 or more is treated as malaria positive.

The original data file has ten patient measurements and the answer column Malaria_Positive. The ten measurements are age, fever temperature, chills, headache severity, vomiting, diarrhoea, muscle pain, fatigue level, sweating and nausea. The data used for training had fifteen features, because five extra features had been added: high fever, fever times chills, severe headache, symptom count and vulnerable age. The values in the training file had also been rescaled so that they are centred around zero, which helps neural networks learn.

The tools used were TensorFlow and Keras to build the network, NumPy and pandas for the data, scikit-learn for the evaluation measures, and Matplotlib and Seaborn for the charts.

## Methodology

The work was done in a Jupyter notebook, in the order below.

**Step 1: Prepare the tools.** The libraries were loaded and random seeds were set to 42 so that the results can be repeated.

**Step 2: Load and separate the data.** Three prepared files were read: a training file with 560 records, a validation file with 120 records and a test file with 120 records. In each file the answer column was separated from the 15 features, and the values were converted to a numeric format the network can use. The shapes were printed to check the sizes. In the notebook the validation set is labelled "Raw" in the printed output, which is a labelling mistake.

**Step 3: Design the network.** The network was built as a function that takes the number of features. It has four stages:
- Block one has 128 neurons, with a small weight penalty (L2 regularisation) to discourage overly large weights, then batch normalisation to keep values stable, then the ReLU activation, then dropout of 30 percent, which switches off random neurons during training so the network cannot rely on a few of them.
- Block two has 32 neurons with batch normalisation, ReLU and dropout of 30 percent.
- Block three has 32 neurons with ReLU and dropout of 20 percent.
- The output is a single neuron with a sigmoid activation, which produces the probability of malaria.

The summary printed in the notebook shows 7,905 parameters in total, of which 7,585 are trainable and 320 belong to the batch-normalisation layers. This is a small network, which suits a small dataset.

**Step 4: A side exercise on matrix shapes.** The notebook contains a short practice cell where a weight matrix, an input and a bias are multiplied by hand. A line that caused a shape error was commented out, and the matrices were transposed so the multiplication worked. The printed result was a four-by-four matrix. This was practice for understanding the shapes inside a network and is not part of the model.

**Step 5: Choose how the model learns.** The model was set up with the Adam optimiser at a low learning rate of 0.00005, the binary cross-entropy loss, which suits yes-or-no problems, and four measures to watch: accuracy, AUC, precision and recall.

**Step 6: Train with safeguards.** The model was allowed up to 300 passes through the data (epochs) in batches of 12 records. Three automatic helpers were used, all based on the validation AUC:
- Early stopping, which ends training if there is no improvement for 25 epochs and restores the best weights.
- Learning-rate reduction, which halves the learning rate if progress stalls for 10 epochs.
- A checkpoint, which saves the best version of the model to a file.

**Step 7: Check the training record.** Training stopped early at epoch 141. The best weights came from epoch 116, and the best validation AUC was 0.9977. In the first epoch the validation accuracy was about 64 percent and the validation AUC about 0.78. By the last epochs the training accuracy was about 97 to 98 percent and the validation accuracy was 96.7 percent, with validation precision and recall both 96 percent. The learning rate was lowered in stages, reaching 0.0000125 by epoch 135. A chart of loss and accuracy over the epochs was saved as an image.

**Step 8: Test on unseen data.** The best saved weights were loaded and the model was evaluated on the 120 test records. The notebook then produced a ROC curve with its AUC, a classification report with precision, recall and F1-score for each class, and a confusion matrix chart. All three were saved as images.

## Discussion

### Results from the saved outputs

| Measure | Result on the test set |
|---|---|
| Loss | 0.0296 |
| Accuracy | 0.9917 (printed under the label compile_metrics, which matches the 0.99 accuracy in the classification report) |
| ROC-AUC | 1.0000 |
| No Malaria: precision, recall, F1 | 0.99, 1.00, 0.99 (70 patients) |
| Malaria: precision, recall, F1 | 1.00, 0.98, 0.99 (50 patients) |

From these numbers, 119 of the 120 test patients were classified correctly. The report implies that the one error was a malaria case predicted as no malaria and that there were no false alarms. This was worked out from the report values and not read from the confusion matrix image.

### What the results mean

The training and validation results are very close to each other, which suggests the network did not memorise the training data. The dropout, the weight penalty and the early stopping appear to have done their job. The test results are as good as the validation results, which is a good sign, because the test set was not used to choose the model.

The choice of AUC to control the early stopping makes sense for a medical problem, because AUC measures how well the model ranks sick patients above healthy ones regardless of the cut-off. The recall for malaria is the most important single measure in practice, since a missed case is more costly than an unnecessary test. Here it was 0.98.

### Why the small results need care

An AUC of exactly 1.0 on only 120 test patients is unusually perfect. It may mean the symptoms in this dataset separate the two groups very clearly, which is more common in simple or generated data than in real clinics. The notebook does not say where the data came from, so this cannot be confirmed. The results should therefore be treated as a successful learning exercise, not as proof that the model would work in a real clinic.

## Challenges

1. The code that created the three data files, added the five extra features and rescaled the values is not in this notebook. The notebook starts from files that were already prepared, so the data preparation cannot be checked or repeated from this notebook alone.
2. The validation set is printed with the label "Raw", and a comment calls the validation data the "raw dataset". This is confusing, because a raw data file also exists in the folder.
3. The third chart in the training-history figure is titled AUC, but it plots accuracy again. The AUC curve was never drawn.
4. In the ROC curve chart, the label for the horizontal axis is commented out, so the axis has no name.
5. In the confusion matrix cell, the horizontal axis label is written as an assignment, which replaces the plotting function with a piece of text instead of setting the label. The chart is therefore missing its predicted-class label, and any later use of that function in the notebook would fail.
6. The test evaluation printed only the loss and one value under the name compile_metrics, so the accuracy, AUC, precision and recall were not shown with proper names in that cell.
7. The matrix practice cell adds a bias to the input in a way that expands the shapes, so its result is not a standard neuron calculation. It is also unrelated to the model and sits in the middle of the notebook.
8. Other leftovers include repeated imports of the same libraries, a text cell that contains code, and cells that were run out of order, shown by the numbering jumping to 76 and then 104.
9. The dataset is small, with 560 training records and 120 each for validation and testing, and the results come from a single split and a single training run.
10. The decision threshold was left at 0.5 without testing alternatives. In a medical setting a lower threshold might be preferred to catch more cases.

## Observations

What is confirmed and what is not:
- The network design, the training settings, the data file contents and every result in this report were verified by reading the notebook and the data files. The results are the outputs saved inside the notebook.
- The notebook was not re-run for this report, so the results were not reproduced.
- The five extra features are named in the data file, but how each one was calculated is not shown, so their definitions are not stated here.
- The source of the data and the number of records in the original raw file were not checked.
- The three chart images in the folder were not opened for this report. Their descriptions come from the notebook code and the printed results.

Observations about the work:
- The model is small, and a small network seems to be enough for this dataset.
- The training record is clean: both losses fall steadily, the two curves stay close, and early stopping ended training at a sensible point.
- The extra engineered features and the scaling probably helped the network learn, but this was not tested by training without them.
- The strongest part of the work is the evaluation, which uses several measures and charts instead of accuracy alone.

## Conclusion

The project builds a small neural network that predicts malaria from ten patient measurements and five added features. It was trained with dropout, a weight penalty, batch normalisation and three automatic helpers, and it was evaluated on a separate test set. The saved results show about 99 percent accuracy, an ROC-AUC of 1.0 and one missed malaria case out of 50 on the test records. These are excellent results for a learning exercise, but the very small test set and the perfect AUC mean they should be treated with care. Before the work is treated as final, the data preparation should be added to the notebook, the chart mistakes corrected, the data source documented, and the model checked with cross-validation or on data from another source.

## GitHub Links
- Project folder: https://github.com/AkongnwiDarius/Generative_Ai/tree/master/1.5_introduction_to_dl/Akongnwi_Darius_MalariaANN_Exercise
- Notebook: https://github.com/AkongnwiDarius/Generative_Ai/blob/master/1.5_introduction_to_dl/Akongnwi_Darius_MalariaANN_Exercise/malaria_prediction.ipynb