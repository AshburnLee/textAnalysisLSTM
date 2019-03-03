import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tqdm import tqdm
import sys
from attention import Attention



'''
read dataset, 
and get the 'comment_text' column,
create a stop_words instance, which is a set()
'''
#stemmer = SnowballStemmer('english')  # did not use stemmer
stop_words = set(stopwords.words("english"))   # set()

# read raw file:  -> DataFrame
train_set = pd.read_csv("./dataset/train.csv")
#train_set.head()
test_set = pd.read_csv("./dataset/test.csv")
#test_set.head()

# get the 'comment column':
#train_text = train_set["comment_text"]
#test_text = test_set["comment_text"]
train_text = train_set.loc[:,"comment_text"]
test_text = test_set.loc[:,"comment_text"]



'''
get the labels from training set: y,
get each comment from all dataset,
'''

# get the comment in ndarrays and clean it:
# Get labels arrays & train array:
labels_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
get_labels = train_set[labels_list]  #-> DataFrame
y = get_labels.values  #-> ndarray -> (159571,6)


train_array = train_set["comment_text"].values  # -> ndarray  (159571,)
test_array = test_set["comment_text"].values   # -> ndarray  (153164,)



'''
create a fucntion to clean all comments,
create two empty lists.
'''
# A list to store cleaned text
train_cleaned_comment_list = [] 
test_cleaned_comment_list = []


# set one peise of comment as para
def clean_comment(comment):
    # remove anything that are not words
    re_train = str(re.sub("[^a-zA-Z]"," ", comment))
    #print(retrain)
    
    # 
    tokened_train = word_tokenize(re_train.lower())
    #print(tokened_train)
    
    # remove stop_words that have not contribution to the meaning
    stoped_train = [_ for _ in tokened_train if _ not in stop_words]
    #print(stoped_train)
    
    # connect words as a string
    stoped_train_str = " ".join(stoped_train) # -> string
    
    return stoped_train_str



'''
Input cleaned data into lists and files
'''
# write to filess
count = 0
#my_file = open('./output/cleaned_train_comment.csv','w')
for _ in tqdm(train_array):
    i = clean_comment(_)
    #print(i)
    train_cleaned_comment_list.append(i)
#    my_file.write(i)
#    my_file.write('\n\n\n')
    count+=1
#my_file.close()
print("The length of the list(train): ",len(train_cleaned_comment_list))
print("The number of commnents(train): ", count)


count = 0
#my_file = open('./output/cleaned_test_comment.csv','w')
for _ in tqdm(test_array):
    i = clean_comment(_)
    #print(i)
    test_cleaned_comment_list.append(i)
#    my_file.write(i)
#    my_file.write('\n\n\n')
    count+=1
#my_file.close()
print("The length of the list(test): ",len(test_cleaned_comment_list))
print("The number of commnents(test): ", count)




'''
For testing: 1000 data for testing


train_cleaned_comment_list = train_cleaned_comment_list[0:1000] 
y = y[0:1000]
test_cleaned_comment_list = test_cleaned_comment_list[0:1000]
'''


'''
vectorization,
extend each comments to the same length, set maxlen=100,
* note that only 1000 of training and testing data are selected.
'''
# vecterization:


tokenizer = Tokenizer(num_words=100000)  

#calculate every appeared words, and index them
tokenizer.fit_on_texts(train_cleaned_comment_list + test_cleaned_comment_list)   

# replace all the words with their index that created from fit_on_texts() 
train_sequences = tokenizer.texts_to_sequences(train_cleaned_comment_list)  
test_sequences = tokenizer.texts_to_sequences(test_cleaned_comment_list)

# for testing
# print(train_cleaned_comment_list)
print(train_sequences[0])  

# every word has an index,
# it is needed for embedding.
word_index = tokenizer.word_index  # len() -> 14121 
print("%d unique words were found"%len(word_index))

# extend or shrink every piece comment to the lenght 100
vec_train_data = pad_sequences(train_sequences, maxlen=100)   # ->(1000, 100)
print("vec_train_data: ",vec_train_data.shape)      

vec_test_data = pad_sequences(test_sequences, maxlen=100)    # ->(1000, 100)
print("vec_test_data: ",vec_test_data.shape)
print("labels: ",y.shape)



"""
split data into training set and validation set
"""

# shuffle all samples:
shuffle_data = np.random.permutation(len(vec_train_data))   #->(1000, )

# get the indeces of train & val:
index_data_train = shuffle_data[: int(len(vec_train_data) * 0.9)]  # -> (900,)
index_data_val = shuffle_data[int(len(vec_train_data) * 0.9) :]    # -> (100,)

# training data  
final_data_train = vec_train_data[index_data_train]  # -> (0.9*, maxlen)           #   (^_^)
labels_train = y[index_data_train]      # labels      #-> (0.9*, 6)                #   (^_^)

#validation data
final_data_val = vec_train_data[index_data_val]     #->(0.1, maxlen)            #  (^_^)
labels_val = y[index_data_val]                     # ->(0.1, 6)                 #  (^_^)



'''
Input train and validation data into and files with their LABELS!!!
'''
# write to files
count = 0
with open('./output/final_train.csv','w') as my_file:
    for _, _1 in zip(final_data_train, labels_train):     # write labels into 'final_train.csv'
        my_file.write(str(_))
        my_file.write(str(_1))
        my_file.write('\n\n\n')
        count+=1
my_file.close()
print("final_train #: ", count) 


count = 0
with open('./output/final_validation.csv','w') as my_file:
    for _, _1 in zip(final_data_val, labels_val):
        my_file.write(str(_))
        my_file.write(str(_1))
        my_file.write('\n\n\n')
        count+=1
my_file.close()
print("final_validation #: ", count)



# What we need from above are :
# vec_test_data            ->(1000,maxlen)
# final_data_train         ->(0.9*, maxlen)
# labels_train             ->(0.9*, 6)
# final_data_val           ->(0.1*, maxlen)
# labels_val               ->(0.1*, 6)




# parameters:
# TRAIN_DATA_FILE = 'train.csv'
# TEST_DATA_FILE = 'test.csv'

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300 
#VALIDATION_SPLIT = 0.9

num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.25
rate_drop_dense = 0.25



# Embedding:
embeddings_index = {}

with open('./glove_matrix/glove.840B.300d.txt','rb') as f:
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs      # -> 2196016
f.close()


# embeddings_index is a dict created by using 'glove_matrix'.
# each 'key' in that dic has a 300D array as its 'value', which is generated by pre-training 'glove'




# then create an embedding_matrix, get the 10,000 words from the begining.
# fill embedding_matrix with 'values',ask 300D arrays.

nb_words = min(MAX_NB_WORDS, len(word_index))   #word_index from previous step
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))   #->(14121+1, 300)

# operation on each <key, value> in dict,
for i, j in tqdm(word_index.items()):
    if j >= MAX_NB_WORDS:     # 100000
        continue
    embedding_vector = embeddings_index.get(str.encode(i))
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[j] = embedding_vector    # shape -> (14121, 300)





# define LSTM layer:
# What is LSTM net:
# To reduce the vanishing (and exploding) gradient problem, 
# and therefore allow deeper networks and recurrent neural networks to perform well in practical settings, 
# there needs to be a way to reduce the multiplication of gradients which are less than zero. 
# The LSTM cell is a specifically designed unit of logic 
# that will help reduce the vanishing gradient problem sufficiently 
# to make recurrent neural networks more useful for long-term memory tasks.

# num_lstm 300
# rate_drop_lstm 0.25
# definition of keras.layers.LSTM()
'''
keras.layers.LSTM(units,                 # output dimention
                  activation='tanh', 
                  recurrent_activation='hard_sigmoid', 
                  use_bias=True, 
                  kernel_initializer='glorot_uniform', 
                  recurrent_initializer='orthogonal', 
                  bias_initializer='zeros', 
                  unit_forget_bias=True, 
                  kernel_regularizer=None, 
                  recurrent_regularizer=None, 
                  bias_regularizer=None, 
                  activity_regularizer=None, 
                  kernel_constraint=None, 
                  recurrent_constraint=None, 
                  bias_constraint=None, 
                  dropout=0.0,           # Fraction of the units to drop for the linear transformation of the inputs.
                  recurrent_dropout=0.0, #  Fraction of the units to drop for the linear transformation of the recurrent state.
                  implementation=1, 
                  return_sequences=False,   #Boolean. Whether to return the last output. in the output sequence, or the full sequence.
                  return_state=False, 
                  go_backwards=False, 
                  stateful=False, 
                  unroll=False)
'''
'''
keras.layers.Embedding(input_dim,      # Size of the vocabulary, i.e. maximum integer index + 1.
                       output_dim,     # Dimension of the dense embedding
                       embeddings_initializer='uniform', 
                       embeddings_regularizer=None, 
                       activity_regularizer=None, 
                       embeddings_constraint=None, 
                       mask_zero=False, 
                       input_length=None)   #Length of input sequences,
'''


'''
define embedding layer & lstm layer:
'''
# embedding_matrix is used here!
embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],    # input_dim = 14121 
                            output_dim=embedding_matrix.shape[1],    # output_dim = 300
                            weights=[embedding_matrix],   # weight -> [(14121,300)]
                            input_length=100,             #
                            trainable=False)

# LSTM layer:
lstm_layer = LSTM(num_lstm,   # 300 
                  dropout=rate_drop_lstm,   # 0.25
                  recurrent_dropout=rate_drop_lstm,
                  return_sequences=True)



'''
Construct our deep net
'''

comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')  #(100,) ->> (?,100)
x = embedding_layer(comment_input)                                  #(?, 100)->> (?, 100, 300)
x = lstm_layer(x)                                                   #(?, 100,300) ->> (?,?,300)
x = Dropout(rate_drop_dense)(x)                                     #(?,?,300) ->> (?,?,300)
x = Attention(MAX_SEQUENCE_LENGTH)(x)                               #(?,?,300) ->> (?,300)
# num_dense = 256
x = Dense(num_dense, activation='relu')(x)                          #(?,300) ->> (?,256)
x = Dropout(rate_drop_dense)(x)                                     #(?,256) ->> (?,256)
x = BatchNormalization()(x)                                         #(?,256) ->> (?,256)
preds = Dense(6, activation='sigmoid')(x)                           #(?,256) ->> (?,6)




# instantiate a Model:
model = Model(inputs=[comment_input],   
              outputs=preds)     # (?,100) ->> (?,6)

# configure the learning process:
model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

print(model.summary())






STAMP = 'simple_lstm_glove_vectors_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense)
print(STAMP)

# define: early_stop, model_checkpoint:
early_stopping =EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = STAMP + '.h5'     # .h5 file
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)






# training : data_train, labels_train, data_val, labels_val!!!
hist = model.fit(final_data_train, labels_train, 
                 validation_data=(final_data_val, labels_val),   # this is validation data
                 epochs=5, 
                 batch_size=256, 
                 shuffle=True,
                 callbacks=[early_stopping, model_checkpoint])

# loads the weights of the model from a HDF5 file (created by 'save_weights'). 
model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])



# test_data 
y_test = model.predict([vec_test_data], batch_size=1024, verbose=1)   # ->(1000, 6)



# before submission, use whiole data seet, 
# submission:
sample_submission = pd.read_csv("./dataset/sample_submission.csv")
sample_submission[labels_list] = y_test    # write out predicted labels into that submission file.

sample_submission.to_csv('%.4f_'%(bst_val_score) + STAMP + '.csv', index=False)    # .csv file










