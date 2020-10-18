'''
LSTM network outperforms the RNN as it can remember long term dependencies(they also overcome the vanishing gradient problem for RNNs).
'''
import tensorflow
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from flask import Flask,render_template,request
from tensorflow.keras.models import model_from_json
from flask_restful import reqparse,Api,Resource

#Flask object instantiation
app=Flask(__name__)
api=Api(app)

model3=model_from_json(open('lstm_model1.json').read())
model3.load_weights('lstm_model1_weights.h5')
model3.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
    
#argument passing
#the parser will look through the parameters that a user sends to your api
parser=reqparse.RequestParser()
parser.add_argument('query')

class Predict(Resource):
    #In get(), we provide directions on how to handle the users query and how to package the JSON object that will be returned to the user
    def get(self):
        #Each class can have several methods that correspond to HTTP methods such as get,put,post and delete.
        #Here get will be the primary method since our main objective is to return predictions
        #using parser to find users query
        args=parser.parse_args()
        text=args['query']
        # df1 consists of 10729 texts 
        df1=pd.read_csv('Sentiment.csv')
        df1=df1[['text','sentiment']]
        df1=df1[df1.sentiment!='Neutral']
        #print(df1)

        label_encoder=LabelEncoder()
        df1['sentiment']=label_encoder.fit_transform(df1['sentiment'])
        #preprocessing data
        for i in range(len(df1.index)):
            text=df1.iloc[i,0]
            text=re.sub('[^a-zA-Z]',' ',text)
            text=(text.lower()).split()
            text=[word for word in text if (word not in set(stopwords.words('english')) and word!='rt')]
            text=' '.join(text)
            #print(text)
            df1.iloc[i,0]=text
        # Vectorizing a text corpus using Tokenizer object
        tokenizer=Tokenizer(num_words=3000,split=' ')  #num_words is the max number of tokens to keep
        # to update internal vocabulary based on a list of texts
        tokenizer.fit_on_texts(df1['text'].values)
        #turning each text into a sequence of integers (each integer being the index of a token in a dictionary)
        X = tokenizer.texts_to_sequences(df1['text'])
        #pad_sequences to convert the sequences into 2-D numpy array.
        X = pad_sequences(X)
    
        #preprocessing input
        list1=[text]
        for input_text in list1:
            input_text=re.sub('[^a-zA-Z]',' ',input_text)
            input_text=(input_text.lower()).split()
            input_text=[word for word in input_text if (word not in set(stopwords.words('english')) and word!='rt')]
            input_text=' '.join(input_text)
        t = tokenizer.texts_to_sequences(list1)
        val = pad_sequences(t,maxlen=24)
        #print("recieved input shape : ",val.shape)
        res=model3.predict_classes(val)
        res=res[0][0]
        #print("RES : ",res)
        if(res==1):
            return {'prediction':1}
        else:
            return {'prediction':0}
        #creating json object
        #returning to user
        
        return output


#Routing the base URL to the resource
api.add_resource(Predict,'/')

#run() makes sure to run only app.py on the server when this script is executed by the Python interpreter

if __name__=='__main__':
    
    #debug==True activates the Flask debugger and provides detailed error messages
    app.run(debug=True)
