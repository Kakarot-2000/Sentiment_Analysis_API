<h1> Sentiment_Analysis </h1>
To perform sentiment analysis on 2016 Presidential Election tweets.
In my previous approach i used Naive Bayes Classifier to solve the problem statement. However LSTMs provide more accuracy than traditional ML approaches( given sufficient data :) ) because of their ability to "remember" i.e to keep track of context in text sequences. Also, an increase in accuracy from 78%(using Naive-Bayes) to 86% was achieved using LSTMs.
<b>Here, I have deployed my model as REST Api so that it can easily be used in different platforms.</b>
Why use LSTM?
LSTMs overcame the vanishing gradients problem in RNN and this resulted in amazing results in speech to text conversion and the advancement of Siri, Cortana, Google voice assistant, Alexa etc. 
They also improved machine translation, which resulted in the ability to translate documents into different languages, translate images into text, text into images, and captioning video etc.

<i>Dataset downloaded from : [link](https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment#)</i>

<h2> Folder Structure:  </h2>
- api/ : contains all the files needed for the flask container
- nginx/ : contains all the files needed for the nginx container
- docker-compose.yml : config file for docker-compose

<h2> How to Run : </h2>
<h3> Make sure that docker is installed on the system before starting </h3>
1. Clone the directory
2. Build the container images (from the directory of the repo)
```
$ docker-compose up --build
```
Note : --build is only needed the first time you are running this command
Now the container will be up and running on http://localhost . </br>
3. Send Request
Open a new terminal and send a POST request
```
$ curl -H "Content-type: application/json" -d '{"_query_":"Enter Your Input Here"}' 'http://localhost:8000/predict'
```

**How to Improve Accuracy :**
One approache is to use an Attention based model like Transformers. RNNs and LSTMs are more difficult to train as compared to Transformers
as they require a lot more memory-bandwidth for computation. Hence they cannot utilize hardware acceleration.
