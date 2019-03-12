#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,Imputer
import random

train = pd.read_csv("Desktop/Pratha/Prac/titanic-mlp-master/train.csv", usecols = ["Pclass","Sex","Age","Parch","Fare"])
train_y = pd.read_csv("Desktop/Pratha/Prac/titanic-mlp-master/train.csv",usecols = ["Survived"])
test = pd.read_csv("Desktop/Pratha/Prac/titanic-mlp-master/test.csv",usecols = ["Pclass","Sex","Age","Parch","Fare"])
test_y = pd.read_csv("Desktop/Pratha/Prac/titanic-mlp-master/gender_submission.csv", usecols = ["Survived"])

for i in np.where(np.isnan(train.iloc[:,2])):
    train.iloc[i,2] = 30

#train['Embarked'] = train['Embarked'].map(lambda x : 0 if x == 'S' else 1 if x =='C' else 2 if x == 'Q' else 0)
labelencoder = LabelEncoder()
#train.iloc[:,6] = labelencoder.fit_transform(train.iloc[:,6])
train.iloc[:,1] = labelencoder.fit_transform(train.iloc[:,1])
onehot = OneHotEncoder(categorical_features = [0,1,3])
train_x = onehot.fit_transform(train.values).toarray()

#one_hot = OneHotEncoder(categorical_features = [0]) 
#train_y = one_hot.fit_transform(train_y.values).toarray()
train_y = train_y.values

#contruction phase
n_inputs = 14
n_hidden1  = 3000
n_hidden2 = 1000
n_outputs = 2
total_size, total_features =train_x.shape

#input output placeholders 
x = tf.placeholder(tf.float32,shape = (None,n_inputs), name = "X")
y = tf.placeholder(tf.int64,shape = (None), name = "Y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(x,n_hidden1,name = "hidden1", activation = tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name ="hidden2", activation = tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name = "outputs")
    
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(xentropy, name ="loss")
    
    
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    opt = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()
    
n_epochs = 40
batch_size = 10

def getnextbatch():
    i = int(random.choice(np.arange(0,total_size - batch_size)))
    x_batch = train_x[i:i+batch_size]
    y_batch = train_y[i:i+batch_size].flatten()
    return x_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(50):
            x_batch,y_batch = getnextbatch()
            sess.run(opt, feed_dict = {x: x_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict = {x: x_batch, y: y_batch})
        print("train acc", acc_train)
    save_path = saver.save(sess,"Desktop/Pratha/Prac/titanic-mlp-master/mymodel_final.ckpt")
    
for i in np.where(np.isnan(test.iloc[:,4])):
    test.iloc[i,4] = 21

test['Parch'] = test['Parch'].map(lambda x : 4 if x == 9 else x )
labelencoder = LabelEncoder()
#test.iloc[:,6] = labelencoder.fit_transform(test.iloc[:,6])
test.iloc[:,1] = labelencoder.fit_transform(test.iloc[:,1])
onehot = OneHotEncoder(categorical_features = [0,1,3])
test_x = onehot.fit_transform(test.values).toarray()     
with tf.Session() as sess:
    saver.restore(sess,"Desktop/Pratha/Prac/titanic-mlp-master/mymodel_final.ckpt")
    x_test_batch = test_x[10:20]
    z = logits.eval(feed_dict = {x:x_test_batch })
    y_preds = np.argmax(z, axis = 1)
        