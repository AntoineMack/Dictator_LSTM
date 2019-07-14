import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pickle

from setup import *

# reshape X to be [samples, time steps, features]
X = np.reshape(data_X_new, (n_patterns_new, seq_length, 1))
# normalize
X = X / float(vocab_count)
# one hot encode the output variable
y = np_utils.to_categorical(data_y_new)

#Build LSTM model
model_new = Sequential()
model_new.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model_new.add(Dropout(0.2))
model_new.add(Dense(y.shape[1], activation='softmax'))
model_new.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint/ save the weights
filepath= "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

epochs = 50  #change number of training epochs here

model_new.fit(X, y, epochs=epochs , batch_size=128, callbacks=callbacks_list)

#Saving the model
filename = 'best_lstm_model.sav'
pickle.dump(model_new, open(filename, 'wb'))
