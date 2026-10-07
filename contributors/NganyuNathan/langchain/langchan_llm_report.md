
# 🧾🧾LANGCHAIN AND LLMS

     `What is an LLm(large language model):` an LLM is a language model is trained on a very huge amount of data and is capable of predicting the next word in a sentence of next sentence in a paragraph. Examples include GPT, Gemini etc.

Important components: 
1) TOKENIZER: it splits text into smaller pieces called tokens so that a neural network can process

2) EMBEDDINGS: embeddings convert tokens into numerical vectors so that the computer can understand 

3) SELF-ATTENTION: it allows the model to determine which parts of the inputs are important in relation to other parts
4) TRASFORMER LAYERS: LLMs contain many layers of attention and neural-network processing.

5) OUTPUT LAYER: produces probabilities for possible next token predictions

#🔑🔐HOW TO CALL AN LLM USING AN API KEY
   - Create an .env file were you can store all your api keys or any important file that is private
   - Go to your browser and search for groq, create an account then go to the get api key on the navbar and create an api key then copy it. Or you can also get an api key from google api keys which allow   you to use to use the gemini model
   - Create a variable eg GOOGLE_API_KEY in your env file copy and paste your key in the new variable eg   GOOGLE_API_KEY = API key


## 🔗🔗LANGCHAIN:  
      lanchain is a framework that provides developers with a common documentation of many models. Langchain helps ease the stress of developers having to read the documentation of every model in order to use them. Instead it find for the most common things in all the models and provide them to the developer
             It provides the developer with componets such as  prompts, model calls, chains, structured outputs, tools, agents, converstional workflows etc. to install langchain you can do pip install langchain on your terminal and it will install all the required packages for langchain    

## Github links
 - langchain: https://github.com/NganyuNathan/up/tree/master/0.7_langchain/1_langchain
