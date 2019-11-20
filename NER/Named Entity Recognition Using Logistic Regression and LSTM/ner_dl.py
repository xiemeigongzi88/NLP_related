# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:33:05 2019

@author: sxw17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import os 
os.getcwd()

path ="C:\\Users\\sxw17\\Desktop\\2019 Fall\\NLP\\assignment\\Project_2\\dl"
os.chdir(path)

df_test = pd.read_csv('test_feature_extraction_step_1.csv')
df_train = pd.read_csv('train_feature_extraction_step_1.csv')


#df_test.columns 

# 46435  sen_3682 
df_test = df_test[['sentence', 'word','ner_tag']]

# 254983 sen_18451
df_train = df_train[['sentence', 'word','ner_tag']]

df = pd.concat([df_train, df_test])

sentences = df['sentence'].tolist()
len(sentences)  # 301418

sen_num = []

for word in sentences:
    num = int(word[4:])
    sen_num.append(num)   
    
sen_num  
len(sen_num) # 301418
len(df)  # 301418


    
    
post_num = []

for i in sen_num[254983:]:
    post_num.append(i+18452)
    
for i in post_num[:10]:
    print(i)

pre_num =[]
for i in sen_num[:len(df_train)]:
    pre_num.append(i)

num = pre_num+post_num 
len(num)

review_sent =[]

for i in num:
    review_sent.append('sentence_'+str(i))
    
review_sent
len(review_sent)

df['sentence']= review_sent

df.reset_index()

##############################################################

df.columns 
# ['sentence', 'word', 'ner_tag']

words = set(df['word'].tolist())
words.add('Padword')
num_words = len(words)

num_words # 26870

tags = set(df['ner_tag'].tolist())
num_tags = len(tags)
tags



agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),s["ner_tag"].values.tolist())]
grouped = df.groupby("sentence").apply(agg_func)
sentences = [s for s in grouped]

largest_sen = max(len(sen) for sen in sentences)
print('biggest sentence has {} words'.format(largest_sen))


    
max_len = 20
X = [[w[0]for w in s] for s in sentences]
new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("PADword")
    new_X.append(new_seq)
new_X[2]



from keras.preprocessing.sequence import pad_sequences
tags2index = {t:i for i,t in enumerate(tags)}
y = [[tags2index[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tags2index["O"])
y[2]


from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
X_tr, X_te, y_tr, y_te = train_test_split(new_X, y, test_size=0.2, random_state=2018)
sess = tf.Session()
K.set_session(sess)
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())



batch_size = 32
def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x,    tf.string)),"sequence_len": tf.constant(batch_size*[max_len])
                     },
                      signature="tokens",
                      as_dict=True)["elmo"]


from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
input_text = Input(shape=(max_len,), dtype=tf.string)
embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
x = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(num_tags, activation="softmax"))(x)
model = Model(input_text, out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


X_tr, X_val = X_tr[:99*batch_size], X_tr[-11*batch_size:]
y_tr, y_val = y_tr[:99*batch_size], y_tr[-11*batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
history = model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),batch_size=batch_size, epochs=3, verbose=1)


####################################################################

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
X_te = X_te[:22*batch_size]
test_pred = model.predict(np.array(X_te), verbose=1)

######################################################
idx2tag = {i: w for w, i in tags2index.items()}
def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PADword", "O"))
        out.append(out_i)
    return out
def test2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p].replace("PADword", "O"))
        out.append(out_i)
    return out
    
pred_labels = pred2label(test_pred)
test_labels = test2label(y_te[:22*32])
print(classification_report(test_labels, pred_labels))






