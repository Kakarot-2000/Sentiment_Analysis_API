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
from nltk.corpus import stopwords
nltk.download('stopwords')
from flask import Flask,render_template,request
from tensorflow.keras.models import model_from_json
from flask_restful import reqparse,Api,Resource
import pickle


df1=pd.read_csv('df2.csv')

# Vectorizing a text corpus using Tokenizer object
tokenizer=Tokenizer(num_words=3000,split=' ')       #num_words is the max number of tokens to keep
tokenizer.fit_on_texts(df1['text'].values)          # to update internal vocabulary based on a list of texts

app=Flask(__name__)                                 #Flask object instantiation
api=Api(app)

model3=model_from_json(open('lstm_model1.json').read())
model3.load_weights('lstm_model1_weights.h5')
model3.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
    
parser=reqparse.RequestParser()                     #the parser will look through the parameters that a user sends to your api
parser.add_argument('query')

class Predict(Resource):
    '''
    In get(), we provide directions on how to handle the users query and how to package the JSON object that will be 
    returned to the user
    '''
    def get(self):
        '''
        Each class can have several methods that correspond to HTTP methods such as get,put,post and delete.
        Here get will be the primary method since our main objective is to return predictions using parser to find users query
        '''
        
        args=parser.parse_args()
        text=args['query']
        
        
        #preprocessing input
        list1=[text]
        for input_text in list1:
            input_text=re.sub('[^a-zA-Z]',' ',input_text)
            input_text=(input_text.lower()).split()
            input_text=[word for word in input_text if (word not in set(stopwords.words('english')) and word!='rt')]
            input_text=' '.join(input_text)
        
        t = tokenizer.texts_to_sequences(list1)
        val = pad_sequences(t,maxlen=24)
        res=model3.predict_classes(val)
        res=res[0][0]
        
        if(res==1):
            return {'prediction':1}
        else:
            return {'prediction':0}

api.add_resource(Predict,'/')           #Routing the base URL to the resource


if __name__=='__main__':    
    app.run(debug=True)                 #debug==True activates the Flask debugger and provides detailed error messages
