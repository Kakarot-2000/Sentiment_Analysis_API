# Sentiment_Analysis
To perform sentiment analysis on tweets
In my previous approach i used Naive Bayes Classifier to solve the problem statement. However LSTMs provide more accuracy than traditional ML approaches( given sufficient data :) ) because of their ability to "remember" i.e to keep track of context in text sequences.

**Here, I have deployed my model as REST Api so that it can easily be used in different platforms.**

Why use LSTM?
LSTMs overcame the vanishing gradients problem in RNN and this resulted in amazing results in speech to text conversion and the advancement of Siri, Cortana, Google voice assistant, Alexa etc. 
They also improved machine translation, which resulted in the ability to translate documents into different languages, translate images into text, text into images, and captioning video etc.

Dataset downloaded from : [link] (https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment#)


##Folder Structure
- app.py: This file contains the Flask specific code.
- lstm_model.py: This file contains the original code used to train the model.

##Ways to Access the API:
- **Using request module in python** 
*Enter this code in jupyter notebook*
```
$import requests
$url='http://127.0.0.1:5000/'
$params={'query':'_enter_input_here_'}
$response=requests.get(url,params) 
$response.json()  
```  
- **Using curl in the terminal**
```
$ curl -X GET http://127.0.0.1:5000/ -d query='_enter_input_here_'
```
