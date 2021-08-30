'''
LSTM network outperforms the RNN as it can remember long term dependencies(they also overcome the vanishing gradient problem for RNNs).
'''

from flask import Flask, jsonify, request
import json
import re
import gunicorn
import pickle
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


df1 = pd.read_csv('df2.csv')

# Vectorizing a text corpus using Tokenizer object
# num_words is the max number of tokens to keep
tokenizer = Tokenizer(num_words=3000, split=' ')
# to update internal vocabulary based on a list of texts
tokenizer.fit_on_texts(df1['text'].values)


app = Flask(__name__)  # Flask object instantiation

model1 = model_from_json(open('lstm_model1.json').read())
model1.load_weights('lstm_model1_weights.h5')
model1.compile(loss='binary_crossentropy',
               optimizer='RMSprop', metrics=['accuracy'])


@app.route('/')
def home():
    return "Hello!!"


@app.route('/predict', methods=['POST'])
def api():
    try:
        json_data = request.get_json()
        list1 = [json_data['_query_']]
        for input_text in list1:  # preprocessing
            input_text = re.sub('[^a-zA-Z]', ' ', input_text)
            input_text = (input_text.lower()).split()
            input_text = [word for word in input_text if (
                word not in set(stopwords.words('english')) and word != 'rt')]
            input_text = ' '.join(input_text)
            print('Working')
            input_text = np.array([input_text])
            Y = tokenizer.texts_to_sequences(pd.Series(input_text))
            val = pad_sequences(Y)
            print(val)
            res = model1.predict(val)
            print('res : ', res)
            res = res[0][0]
            if(res >= 0.5):
                return jsonify(predictions={'res': 1})
            else:
                return jsonify(predictions={'res': 0})
    except Exception as e:
        responses = jsonify(predictions={'error': e})
        responses.status_code = 404
    return (responses)


app.run()
