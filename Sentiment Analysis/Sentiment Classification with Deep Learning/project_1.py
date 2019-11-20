# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 00:53:50 2019

@author: sxw17
"""

import os 
os.getcwd()
import pandas as pd
import string

from nltk.corpus import stopwords
import string
import re


path ="C:\\Users\\sxw17\\Desktop\\2019 Fall\\NLP\\assignment\\project_1_dl\\movie_reviews"

os.chdir(path)

# Task 1 ;  Load the data 
'''
1.1 read file(path to dataset): This method takes in the path to a tar
file and produces two lists, one containing movie reviews and the other
containing classes.
'''
def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]'%re.escape(string.punctuation))
    
    tokens=[re_punc.sub('',w) for w in tokens]
    
    tokens = [word for word in tokens if word.isalpha()]
    
    stop_words = set(stopwords.words('english'))
    
    tokens=[w for w in tokens if not w in stop_words]
    
    tokens = [word for word in tokens if len(word) >1]
    
    return tokens


def read_file(path):

    neg_path = path +'\\neg'
    pos_path = path +'\\pos'
    
    neg_files= os.listdir(neg_path) #得到文件夹下的所有文件名称
    neg_reviews = []
    for file in neg_files: #遍历文件夹
         if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
              f = open(neg_path+"\\"+file); #打开文件
              iter_f = iter(f); #创建迭代器
              str = ""
              for line in iter_f: #遍历文件，一行行遍历，读取文本
                  str = str + line
              content = clean_doc(str)
              
              neg_reviews.append((content, 'NEGATIVE')) #每个文件的文本存到list中
    
    neg_df = pd.DataFrame(neg_reviews, columns=['Data','Labels'])
    
    
    pos_files= os.listdir(pos_path) #得到文件夹下的所有文件名称
    pos_reviews = []
    for file in pos_files: #遍历文件夹
         if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
              f = open(pos_path+"\\"+file); #打开文件
              iter_f = iter(f); #创建迭代器
              str = ""
              for line in iter_f: #遍历文件，一行行遍历，读取文本
                  str = str + line
                  
              content = clean_doc(str)
              pos_reviews.append((content, 'POSITIVE')) #每个文件的文本存到list中
    
    pos_df = pd.DataFrame(pos_reviews, columns=['Data','Labels'])
    
    res = pd.concat([pos_df, neg_df], axis=0)

    return res

file_output = read_file(path)
file_output.reset_index(drop=True)

########################################################################
'''
1.2 preprocess(text): This method creates a vocabulary and associates the
tokens with a unique integer.
'''

total_contents = []

total_reviews = file_output['Data'].tolist()

for each in total_reviews:
    total_contents.append(each)
    
total_contents

def preprocess(total_contents):   
    
    each_word_set=set()
    for i in total_contents:
        for j in i:
            each_word_set.add(j)
            
    
    each_word_list=list(each_word_set) # 46557
    
    vocabulary_dic={}
    
    for i, word in enumerate(each_word_list):
        vocabulary_dic[word]=i
        
    return vocabulary_dic

token_number = preprocess(total_contents)

##############################################
'''
1.3. encode review(vocab, text): Returns integer-encoded movie reviews.
'''
def encode_review(vocab, df):
    
    encode_list = []
    
    for review in df['Data'].tolist():
        
        review_word =[]
        for word in review:
            
            if word in vocab.keys():
                review_word.append(vocab[word])
                
        encode_list.append(review_word)
        
    encode_list
    
    if len(encode_list)== len(df):
        return encode_list
    
encode_list = encode_review(token_number, file_output)

###################################################
'''
1.4 encode labels(labels): Here you integer-encode the labels if you did
not do that while reading the files.
'''

import pandas as pd 

def encode_labels(encode_list, file_output):
    col_1 = encode_list
    judgement = file_output['Labels'].tolist()
    col_2=[]
    
    for i in judgement:
        if i=='POSITIVE':
            col_2.append(1)
        else:
            col_2.append(0)
    
    df=[]
    
    for i in range(len(col_1)):
        df.append((col_1[i],col_2[i]))
        
    data_df = pd.DataFrame(df, columns=['Data', 'Label'])
    
    return data_df

encode_df= encode_labels(encode_list, file_output)

################################################
'''
1.5 pad zeros(encoded reviews, seq length): Here you pad zeros to each
review to make all reviews have equal length. 
'''
max_length=0 

for review in file_output['Data'].tolist():
    if max_length< len(review):
        max_length=len(review)
        
max_length # 1380 


def get_max_length(df):
    
    max_length = 0 
    
    for i in df.Data.tolist():
        if len(i)> max_length:
            max_length= len(i)
        
    return max_length 

seq_length = get_max_length(encode_df)  # 1380 


def pad_zeros( encode_df,seq_length):
    data = encode_df.Data.tolist()
    pad = []
    temp=[]
    
    for i, review in enumerate(data):
        temp = review + [0]*max_length
        temp= temp[:max_length]
        
        pad.append((temp, encode_df['Label'].tolist()[i]))
        
    
    df = pd.DataFrame(pad, columns=['Data','Label'])
    return df

final_df = pad_zeros(encode_df, max_length)

token_number


####################################################################
# Task 2:Build the Embedding dictionary

from gensim.models import Word2Vec
# define training data
sentences = file_output['Data'].tolist()

import torch
from torch.autograd import Variable

def load_embedding_file(file_output, token_dict):
    
    sentences = file_output['Data'].tolist()
    # train model
    model = Word2Vec(sentences, min_count=1)
    # summarize vocabulary
    words = list(model.wv.vocab)

    embedding_dic = {}

    reverse_token ={v: k for k, v in token_number.items()}
    for key, value in reverse_token.items():
        vector = model.wv[value]
        numpy_tensor = Variable(torch.from_numpy(vector))
        embedding_dic[key] = numpy_tensor
        
    return embedding_dic


embedding_dic = load_embedding_file(file_output, token_number) # 46557

#############################################
# Task3&4:  Create a TensorDataset and DataLoader & Define the baseline model 

total_sentences=[]
model = Word2Vec(sentences, min_count=1)
for sentence in sentences:
    #print(sentence)
    each_sentence={}
    for word in sentence:
        vector = model.wv[word]
        numpy_tensor = Variable(torch.from_numpy(vector))
        each_sentence[word]= numpy_tensor
        
    total_sentences.append(each_sentence)
    
total_sentences

import pandas as pd 


label = final_df['Label'].tolist()
df_list=[]

for i in range(len(total_sentences)):
    
    temp_tensor = total_sentences[i]
    temp_label = label[i]
    
    df_list.append((temp_tensor, temp_label))
    
df = pd.DataFrame(df_list, columns=['review', 'label'])
    
 
df

###########################################

from keras.layers import Dropout, Dense, GRU, Embedding
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
from keras.layers import Dense, Dropout, Activation

from sklearn.model_selection import train_test_split
x_pre_train, x_pre_test, y_pre_train, y_pre_test = train_test_split(final_df['Data'], final_df['Label'], test_size = 0.2)

y_train = np.array(y_pre_train)
y_test = np.array(y_pre_test)


temp_test_list = x_pre_test.tolist()
temp_x_test = np.array((temp_test_list[0]))
for i in temp_test_list[1:]:
    #print(i)
    
    temp_x_test = np.vstack((temp_x_test, np.array(i)))
    #print(len(temp_x_test))
    
    
x_test = temp_x_test  # (400, 1380)


temp_train_list = x_pre_train.tolist()
temp_x_train = np.array((temp_train_list[0]))
for i in temp_train_list[1:]:
    #print(i)
    
    temp_x_train = np.vstack((temp_x_train, np.array(i)))
    print(len(temp_x_train))
     
x_train = temp_x_train  # (1600, 1380)


reviews_list = df['review'].tolist() # 2000 

total_dict ={}  # 46557

for i in reviews_list:
    temp_dic = i 
    
    for key, value in temp_dic.items():
        #print(key, value)
        total_dict[key]= value
        

total_dict

##########################################################################
# Task 5: 
# vinilla 
def Build_Model_Vanilla_RNN_Text(word_index, embeddings_index, nclasses=1, MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=100):
    
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                 "into shape",str(len(embedding_vector))," Please make sure your"
                                 " EMBEDDING_DIM is equal to embedding_vector file ,")
                exit(1)

            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))
    model.add(Dropout(0.25))
    
    
    model = Sequential()
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nclasses, activation='softmax'))
    
    
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model


model_vanilla_RNN = Build_Model_Vanilla_RNN_Text(token_number,total_dict)
model_vanilla_RNN.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=100,
                              batch_size=32)


##################################################
# LSTM  ok 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM



def Build_Model_RNN_LSTM_Text(word_index, embeddings_index, nclasses=1, MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=100):

    
    kernel_size = 2
    filters = 256
    pool_size = 2
    gru_node = 256

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                 "into shape",str(len(embedding_vector))," Please make sure your"
                                 " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)

            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(gru_node, recurrent_dropout=0.2))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


model_RNN_LSTM = Build_Model_RNN_LSTM_Text(token_number,total_dict)

model_RNN_LSTM.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=50,
                              batch_size=32)


###################################################
#GRU  ok 
def Build_Model_RNN_GRU_Text(word_index, embeddings_index, nclasses=1,  MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=100, dropout=0.5):

    model = Sequential()
    hidden_layer = 2
    gru_node = 32

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))

    
    print(gru_node)
    model.add(Dropout(0.25))

    model.add(Dense(nclasses, activation='sigmoid'))
    
    
    for i in range(0,hidden_layer):
        model.add(GRU(gru_node,return_sequences=True, recurrent_dropout=0.2))
        model.add(Dropout(dropout))
        
    model.add(GRU(gru_node, recurrent_dropout=0.2))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nclasses, activation='softmax'))


    model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    return model


model_RNN_GRU = Build_Model_RNN_GRU_Text(token_number,total_dict)

model_RNN_GRU.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=20,
                              batch_size=32)

###################################################################

# Bidirectional  
'''
RNN Bidirectional
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
'''
from keras.layers import Bidirectional
from keras import optimizers

def Build_Model_RNN_LSTM_Text_bi(word_index, embeddings_index, nclasses=1, MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=100):

    
    kernel_size = 2
    filters = 256
    pool_size = 2
    gru_node = 256

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                 "into shape",str(len(embedding_vector))," Please make sure your"
                                 " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)

            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Bidirectional(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2)))
    model.add(Bidirectional(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2)))
    model.add(Bidirectional(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2)))
    model.add(Bidirectional(LSTM(gru_node, recurrent_dropout=0.2)))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


model_RNN_LSTM_bi = Build_Model_RNN_LSTM_Text_bi(token_number,total_dict)

model_RNN_LSTM_bi.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=50,
                              batch_size=32)

################################################
# Keras Learning rate adjustment


#################################################
# Task 6: Replace the RNN by self-attention

from keras_self_attention import SeqSelfAttention

def Build_Model_Attention_RNN_LSTM_Text(word_index, embeddings_index, nclasses=1, MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=100):

    
    kernel_size = 2
    filters = 256
    pool_size = 2
    gru_node = 256

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                 "into shape",str(len(embedding_vector))," Please make sure your"
                                 " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)

            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Bidirectional(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Bidirectional(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Bidirectional(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Bidirectional(LSTM(gru_node, recurrent_dropout=0.2)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


model_RNN_LSTM_Attention = Build_Model_Attention_RNN_LSTM_Text(token_number,total_dict)

model_RNN_LSTM_Attention.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=50,
                              batch_size=32)


############################################################
#GRU  ok 
def Build_Model_Attention_RNN_GRU_Text(word_index, embeddings_index, nclasses=1,  MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=1380, dropout=0.5):

    model = Sequential()
    hidden_layer = 2
    gru_node = 32

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))

    
    print(gru_node)
    model.add(Dropout(0.25))

    model.add(Dense(nclasses, activation='sigmoid'))
    
    
    for i in range(0,hidden_layer):
        model.add(GRU(gru_node,return_sequences=True, recurrent_dropout=0.2))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(Dropout(dropout))
        
    model.add(GRU(gru_node, recurrent_dropout=0.2))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dense(nclasses, activation='softmax'))


    model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    return model


model_RNN_GRU_Attention = Build_Model_Attention_RNN_GRU_Text(token_number,total_dict)

model_RNN_GRU_Attention.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=50,
                              batch_size=32)


#################################################################
# Task 7: Train and test the model
# vinilla 
def Build_Model_Vanilla_RNN_Text(word_index, embeddings_index, nclasses=1, MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=100):
    
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                 "into shape",str(len(embedding_vector))," Please make sure your"
                                 " EMBEDDING_DIM is equal to embedding_vector file ,")
                exit(1)

            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))
    model.add(Dropout(0.25))
    
    
    model = Sequential()
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nclasses, activation='softmax'))
    
    
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model


model_vanilla_RNN = Build_Model_Vanilla_RNN_Text(token_number,total_dict)
model_vanilla_RNN.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=100,
                              batch_size=32)

##########################################################################
# LSTM 
def Build_Model_RNN_LSTM_Text(word_index, embeddings_index, nclasses=1, MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=100):

    
    kernel_size = 2
    filters = 256
    pool_size = 2
    gru_node = 256

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                 "into shape",str(len(embedding_vector))," Please make sure your"
                                 " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)

            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(gru_node, recurrent_dropout=0.2))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


model_RNN_LSTM = Build_Model_RNN_LSTM_Text(token_number,total_dict)

model_RNN_LSTM.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=50,
                              batch_size=32)

##############################################################
#GRU  ok 
def Build_Model_RNN_GRU_Text(word_index, embeddings_index, nclasses=1,  MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=1380, dropout=0.5):

    model = Sequential()
    hidden_layer = 2
    gru_node = 32

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))

    
    print(gru_node)
    model.add(Dropout(0.25))

    model.add(Dense(nclasses, activation='sigmoid'))
    
    
    for i in range(0,hidden_layer):
        model.add(GRU(gru_node,return_sequences=True, recurrent_dropout=0.2))
        model.add(Dropout(dropout))
        
    model.add(GRU(gru_node, recurrent_dropout=0.2))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nclasses, activation='softmax'))


    model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    return model


model_RNN_GRU = Build_Model_RNN_GRU_Text(token_number,total_dict)

model_RNN_GRU.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=20,
                              batch_size=32)


########################################################
# change parameters 

def Build_Model_RNN_GRU_Text_cha_P(word_index, embeddings_index, nclasses=1,  MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=100, dropout=0.5):

    model = Sequential()
    hidden_layer = 3 # adding one layer 
    gru_node = 64 # 32-> 64 

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))

    
    print(gru_node)
    model.add(Dropout(0.25))

    model.add(Dense(nclasses, activation='sigmoid'))
    
    
    for i in range(0,hidden_layer):
        model.add(GRU(gru_node,return_sequences=True, recurrent_dropout=0.2))
        model.add(Dropout(dropout))
        
    model.add(GRU(gru_node, recurrent_dropout=0.2))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu'))  # 256->128
    model.add(Dense(nclasses, activation='softmax'))


    model.compile(loss='binary_crossentropy',
                      optimizer='sgd',  # 'rmsprop'->'sgd'
                      metrics=['accuracy'])
    return model


model_RNN_GRU_cha_P = Build_Model_RNN_GRU_Text_cha_P(token_number,total_dict)

model_RNN_GRU_cha_P.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=30,
                              batch_size=64)  # 64->32 


#####################################################################
# Task8: CNN exploration 
# CNN  ok 
from keras.layers import Dropout, Dense,Input,Embedding,Flatten, MaxPooling1D, Conv1D
from keras.models import Sequential,Model
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
from keras.layers.merge import Concatenate


def Build_Model_CNN_Text(word_index, embeddings_index, nclasses=2, MAX_SEQUENCE_LENGTH=1380, EMBEDDING_DIM=100, dropout=0.5):

    """
        def buildModel_CNN(word_index, embeddings_index, nclasses, MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=50, dropout=0.5):
        word_index in word index ,
        embeddings_index is embeddings index, look at data_helper.py
        nClasses is number of classes,
        MAX_SEQUENCE_LENGTH is maximum lenght of text sequences,
        EMBEDDING_DIM is an int value for dimention of word embedding look at data_helper.py
    """

    model = Sequential()
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) !=len(embedding_vector):
                print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                 "into shape",str(len(embedding_vector))," Please make sure your"
                                 " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)

            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    # applying a more complex convolutional approach
    convs = []
    filter_sizes = []
    layer = 5
    print("Filter  ",layer)
    for fl in range(0,layer):
        filter_sizes.append((fl+2))

    node = 128
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(node, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        #l_pool = Dropout(0.25)(l_pool)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1)(convs)
    l_cov1 = Conv1D(node, 5, activation='relu')(l_merge)
    l_cov1 = Dropout(dropout)(l_cov1)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(node, 5, activation='relu')(l_pool1)
    l_cov2 = Dropout(dropout)(l_cov2)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(1024, activation='relu')(l_flat)
    l_dense = Dropout(dropout)(l_dense)
    l_dense = Dense(512, activation='relu')(l_dense)
    l_dense = Dropout(dropout)(l_dense)
    preds = Dense(1, activation='softmax')(l_dense)
    model = Model(sequence_input, preds)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

model_RNN = Build_Model_CNN_Text(token_number,total_dict)

model_RNN.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=10,
                              batch_size=32)

