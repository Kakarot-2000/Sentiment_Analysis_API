'''
LSTM network outperforms the RNN as it can remember long term dependencies(they also overcome the vanishing gradient problem for RNNs).
'''

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import tensorflow
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import re
import nltk
nltk.download('stopwords')


df1 = pd.read_csv('Sentiment.csv')                # df1 consists of 10729 texts
df1 = df1[['text', 'sentiment']]
df1 = df1[df1.sentiment != 'Neutral']
print(df1)


# preprocessing data
for i in range(len(df1.index)):
    text = df1.iloc[i, 0]
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = (text.lower()).split()
    text = [word for word in text if (word not in set(
        stopwords.words('english')) and word != 'rt')]
    text = ' '.join(text)
    df1.iloc[i, 0] = text

# Vectorizing a text corpus using Tokenizer object
# num_words is the max number of tokens to keep
tokenizer = Tokenizer(num_words=3000, split=' ')

# to update internal vocabulary based on a list of texts
tokenizer.fit_on_texts(df1['text'].values)

# turning each text into a sequence of integers (each integer being the index of a token in a dictionary)
X = tokenizer.texts_to_sequences(df1['text'])

# pad_sequences to convert the sequences into 2-D numpy array.
X = pad_sequences(X)


model1 = keras.models.Sequential()

'''
The embedding layer encodes the input sequence into a sequence of dense vectors of dimension 128.
The LSTM layer transforms the vector sequence into a single vector of size 100, containing information about the entire sequence.
'''

model1.add(keras.layers.Embedding(input_dim=15000,
           output_dim=128, input_length=X.shape[1], dropout=0.2))
model1.add(keras.layers.LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
model1.add(keras.layers.Dense(1, activation='relu',
           kernel_initializer='glorot_uniform'))
model1.compile(loss='binary_crossentropy',
               optimizer='RMSprop', metrics=['accuracy'])


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df1['sentiment'])


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100)


early_stopping = keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint('chkpnt1.h5', save_best_only=True)
hist = model1.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs=100, batch_size=32, callbacks=[early_stopping, checkpoint])


file1 = model1.to_json()  # to save model architecture and weights


with open('lstm_model1.json', 'w') as json_file:
    json_file.write(file1)

model1.save_weights('lstm_model1_weights.h5')


#import matplotlib.pyplot as plt
# print(model1.evaluate(x_test,y_test))
# plt.plot(hist.history['accuracy'])
