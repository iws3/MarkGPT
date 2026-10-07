# 🧾🧾RNN (Recurrent Neural Network)

- what is an RNN:
     An RNN is process of making computers to understand text and process text. Or it is the combination of two or more artificial neural networks. RNNs are needed because the problem with classical machine learning is that when the dataset is large, with large number of features, manual feature engineering is tough and time consuming. 

  ## Why RNNs over ANNs
   - Image recognition: this is because an image has tomany inputs which is too much for an ANN to handle so we RNNs instead
   - Language data: this is because CNNs and ANNs cannot work with sequencial data
   - No memory: ANNs do not have memory so we need RNNs to preserve memory for the model

 # Types of RNNs

 1) *one-to-one: 
      Standard neural network. One input, one output. not really sequential, but its the baseline.
 2) *one-to-many:
      One input, many outputs. example image captioning you feed it a whole sentence(many words), it outputs one sentiment score- positive or nagative
 3) *many-to-one:
      Many input, many outputs. this has two subtypes