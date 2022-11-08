1.Import requried library

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras_preprocessing import sequence
from keras.utils import to_categorical
from keras.models import load_model
import csv
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')  
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
2.Read dataset and do preprocessing

from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
cd/content/drive/MyDrive/Colab Notebooks
/content/drive/MyDrive/Colab Notebooks
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/spam.csv',delimiter=',',encoding='latin-1')
df.head()
v1	v2	Unnamed: 2	Unnamed: 3	Unnamed: 4
0	ham	Go until jurong point, crazy.. Available only ...	NaN	NaN	NaN
1	ham	Ok lar... Joking wif u oni...	NaN	NaN	NaN
2	spam	Free entry in 2 a wkly comp to win FA Cup fina...	NaN	NaN	NaN
3	ham	U dun say so early hor... U c already then say...	NaN	NaN	NaN
4	ham	Nah I don't think he goes to usf, he lives aro...	NaN	NaN	NaN
df.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True) #dropping unwanted columns
df.info()
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   v1      5572 non-null   object
 1   v2      5572 non-null   object
dtypes: object(2)
memory usage: 87.2+ KB
#Count of Spam and Ham values
df.groupby(['v1']).size()
v1
ham     4825
spam     747
dtype: int64
#Label Encoding target column
X = df.v2
Y = df.v1
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
# Test and train spilit 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
# Tokenisation function
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
Create Model
Add layers (LSTM ,Dense-(Hidden Layers),Ouput)
#creating LSTM model
inputs = Input(name='InputLayer',shape=[max_len])
layer = Embedding(max_words,50,input_length=max_len)(inputs)
layer = LSTM(64)(layer)
layer = Dense(256,name='FullyConnectedLayer1')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(1,name='OutputLayer')(layer)
layer = Activation('sigmoid')(layer)
5. Compile the model

model = Model(inputs=inputs,outputs=layer)
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 InputLayer (InputLayer)     [(None, 150)]             0         
                                                                 
 embedding (Embedding)       (None, 150, 50)           50000     
                                                                 
 lstm (LSTM)                 (None, 64)                29440     
                                                                 
 FullyConnectedLayer1 (Dense  (None, 256)              16640     
 )                                                               
                                                                 
 activation (Activation)     (None, 256)               0         
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 OutputLayer (Dense)         (None, 1)                 257       
                                                                 
 activation_1 (Activation)   (None, 1)                 0         
                                                                 
=================================================================
Total params: 96,337
Trainable params: 96,337
Non-trainable params: 0
_________________________________________________________________
6. Fit the model

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2)
Epoch 1/10
30/30 [==============================] - 11s 281ms/step - loss: 0.3331 - accuracy: 0.8714 - val_loss: 0.1700 - val_accuracy: 0.9778
Epoch 2/10
30/30 [==============================] - 8s 262ms/step - loss: 0.0897 - accuracy: 0.9768 - val_loss: 0.0778 - val_accuracy: 0.9778
Epoch 3/10
30/30 [==============================] - 8s 260ms/step - loss: 0.0440 - accuracy: 0.9886 - val_loss: 0.0600 - val_accuracy: 0.9821
Epoch 4/10
30/30 [==============================] - 8s 262ms/step - loss: 0.0349 - accuracy: 0.9897 - val_loss: 0.0648 - val_accuracy: 0.9842
Epoch 5/10
30/30 [==============================] - 8s 260ms/step - loss: 0.0249 - accuracy: 0.9918 - val_loss: 0.0761 - val_accuracy: 0.9736
Epoch 6/10
30/30 [==============================] - 8s 261ms/step - loss: 0.0192 - accuracy: 0.9942 - val_loss: 0.0668 - val_accuracy: 0.9863
Epoch 7/10
30/30 [==============================] - 8s 262ms/step - loss: 0.0135 - accuracy: 0.9960 - val_loss: 0.0849 - val_accuracy: 0.9863
Epoch 8/10
30/30 [==============================] - 8s 261ms/step - loss: 0.0094 - accuracy: 0.9976 - val_loss: 0.0862 - val_accuracy: 0.9821
Epoch 9/10
30/30 [==============================] - 8s 264ms/step - loss: 0.0072 - accuracy: 0.9979 - val_loss: 0.0974 - val_accuracy: 0.9831
Epoch 10/10
30/30 [==============================] - 10s 337ms/step - loss: 0.0059 - accuracy: 0.9989 - val_loss: 0.1132 - val_accuracy: 0.9831
Save the model
model.save("model_1")
WARNING:absl:Function `_wrapped_model` contains input name(s) InputLayer with unsupported characters which will be renamed to inputlayer in the SavedModel.
WARNING:absl:Found untraced functions such as lstm_cell_layer_call_fn, lstm_cell_layer_call_and_return_conditional_losses while saving (showing 2 of 2). These functions will not be directly callable after loading.
Test the Model
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accuracy = model.evaluate(test_sequences_matrix,Y_test)
print('Accuracy: {:0.3f}'.format(accuracy[1]))
27/27 [==============================] - 1s 23ms/step - loss: 0.0857 - accuracy: 0.9844
Accuracy: 0.984
y_pred = model.predict(test_sequences_matrix)
print(y_pred[25:40].round(3))
27/27 [==============================] - 1s 22ms/step
[[0.   ]
 [0.   ]
 [1.   ]
 [0.   ]
 [0.967]
 [0.   ]
 [0.   ]
 [0.   ]
 [1.   ]
 [0.003]
 [0.   ]
 [0.   ]
 [0.   ]
 [1.   ]
 [0.   ]]
print(Y_test[25:40])
[[0]
 [0]
 [1]
 [0]
 [1]
 [0]
 [0]
 [0]
 [1]
 [0]
 [0]
 [0]
 [0]
 [1]
 [0]]
