import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import re as re
import sys
import os

#I'm using Keras for simplicity
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#variable needed from train_lstm.py
from load_lstm import *
from setup import *

# global loaded_model
# loaded_model = init()

# User inputs a seed for model to generate predictions
seed = input("Enter your seed sentence ")
seed_list = re.sub("[^\w]", " ",  seed).split()

#seed_list to intergers
toke_seed_list = [word_to_index[words] for words in seed_list]
pattern = toke_seed_list

print("Seed")
#This step is not necessary in case of input seed
print("\"", ' '.join([index_to_word[value] for value in pattern]), "\"") #

#Generate text from corpus based on probability, ie Context
for i in range(20): #We adjust the length of the text here
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_count)
    prediction = loaded_model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = index_to_word[index]
    sequ_in = [index_to_word[value] for value in pattern]
    sys.stdout.write(result + " ")
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")
