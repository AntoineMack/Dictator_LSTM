import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from keras.utils import to_categorical

sp = pd.read_csv('dictator_speech_utf8.csv') #add direct path to any corpus here

sp["text"] = sp["text"].str.replace('[^a-zA-Z]', ' ')   # Use the .str.replace on the SERIES.
sp['text'] = [i.lower() for i in sp['text']]  #lowercase at characters

# Turning corpus into one list for lstm to scan
all_text_to_list = []
for i in sp['text']:
    all_text_to_list.append(i)

# Tokenizing the corpus. Each word has been given an integer value
def tokenize_and_process(text, vocab_size=10000):
    #list for clean text from above
    #all_text_to_list

    T = Tokenizer(num_words = vocab_size)

    #fit tokenizer
    T.fit_on_texts(all_text_to_list)

    #turn input text into sequence of integers
    data = T.texts_to_sequences(all_text_to_list)

    #extract vocabulary word/ index pairings from tokenizer, so we can go back and forth
    word_to_index = T.word_index
    index_to_word = {v: k for k, v in word_to_index.items()}

    return data, word_to_index, index_to_word, T

#Drawing values from above function tokenize_and_process
data, word_to_index, index_to_word, T = tokenize_and_process(all_text_to_list)

#Combine all strings in a list
big_string_list = " ".join(all_text_to_list)

#break giant string into list items
big_text_list = big_string_list.split()

# Total number of words in the corpus, stop words are not removed
count= []
for i in data:
    for j in i:
        count.append(j)
word_count = len(count)
print(word_count)

#total number of unique words in the corpus
vocab_count = len(word_to_index)
print(vocab_count)

# Prepare data for LSTM
all_int_to_list = []         #data list of list containing intergers, has been turned into one large list
for i in data:               # of integers from the corpus.  LSTM will need to scan the entire corpus not rows
    all_int_to_list.append(i)

seq_length = 7
data_X_new = []
data_y_new = []
for i in range(0, word_count - seq_length, 1):
    seq_in = big_text_list[i:i + seq_length]
    seq_out = big_text_list[i + seq_length]
    data_X_new.append([word_to_index[words] for words in seq_in])
    data_y_new.append(word_to_index[seq_out])
n_patterns_new = len(data_X_new)
print("Total Patterns: ", n_patterns_new)
