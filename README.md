# Sentiment_Analysis
To perform sentiment analysis on tweets
In my previous approach i used Naive Bayes Classifier to solve the problem statement. However LSTMs provide more accuracy than traditional ML approaches( given sufficient data :) ) because of their ability to "remember" i.e to keep track of context in text sequences.

Why use LSTM?
LSTMs overcame the vanishing gradients problem in RNN and this resulted in amazing results in speech to text conversion and the advancement of Siri, Cortana, Google voice assistant, Alexa etc. 
They also improved machine translation, which resulted in the ability to translate documents into different languages, translate images into text, text into images, and captioning video etc.

Dataset downloaded from : https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment#


Folder Structure
1. app.py: This file contains the Flask specific code.
2. Templates folder: This folder contains the HTML files. These HTML files will be rendered on the web browser.
3. Styles folder: This folder contains CSS files.

To Run The Application
1. Run 'python app.py' on cmd
   This will now run your python application using Flask micro framework on your local machine. 
2. Enter the local host http address into browser.
