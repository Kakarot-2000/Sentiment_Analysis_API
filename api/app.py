'''
LSTM network outperforms the RNN as it can remember long term dependencies(they also overcome the vanishing gradient problem for RNNs).
'''

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from flask import Flask,jsonify,request
from tensorflow.keras.models import model_from_json
import pickle
import gunicorn
import re
import json


df1=pd.read_csv('df2.csv')

# Vectorizing a text corpus using Tokenizer object
tokenizer=Tokenizer(num_words=3000,split=' ')       #num_words is the max number of tokens to keep
tokenizer.fit_on_texts(df1['text'].values)          # to update internal vocabulary based on a list of texts


app=Flask(__name__)                                 #Flask object instantiation

model3=model_from_json(open('lstm_model1.json').read())
model3.load_weights('lstm_model1_weights.h5')
model3.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])

@app.route('/')
def home():
    return "Hello!!"

@app.route('/predict',methods=['POST'])
def api():
    try:
        json_data = request.get_json()
        list1=[json_data['_query_']]
        for input_text in list1:                           #preprocessing
            input_text=re.sub('[^a-zA-Z]',' ',input_text)
            input_text=(input_text.lower()).split()
            input_text=[word for word in input_text if (word not in set(stopwords.words('english')) and word!='rt')]
            input_text=' '.join(input_text)
        print('Working')
        t = tokenizer.texts_to_sequences(list1)
        val = pad_sequences(t,maxlen=24)
        res=model3.predict_classes(val)
        print('res : ',res)
        res=res[0][0]
        if(res==1):
            return jsonify(predictions={'res':1})
        else:
            return jsonify(predictions={'res':0})
    except Exception as e:
        responses = jsonify(predictions={'error':e})
        responses.status_code = 404
    return (responses)
