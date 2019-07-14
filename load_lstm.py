#change number of epochs in filename

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pickle


print('Starting Load')
def init():
    filename = 'best_lstm_model.sav'
    weights = "weights-improvement-50-3.9219.hdf5"
    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_model.load_weights(weights)
    loaded_model.compile(loss= 'categorical_crossentropy', optimizer='adam')

    return loaded_model

print('Loaded Model from disk')
