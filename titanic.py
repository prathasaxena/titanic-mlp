# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:30:13 2019

@author: Shashank_Saxena
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,Imputer
import random
%matplotlib inline

train = pd.read_csv(r"Titanic\train.csv", usecols = ["Pclass","Sex","Age","Parch","Fare"])
train_y = pd.read_csv(r"Titanic\train.csv",usecols = ["Survived"])
test = pd.read_csv(r"Titanic\test.csv",usecols = ["Pclass","Sex","Age","Parch","Fare"])
test_y = pd.read_csv(r"Titanic\gender_submission.csv", usecols = ["Survived"])


#data processing
for i in np.where(np.isnan(train.iloc[:,2])):
    train.iloc[i,2] = 30

#train['Embarked'] = train['Embarked'].map(lambda x : 0 if x == 'S' else 1 if x =='C' else 2 if x == 'Q' else 0)
labelencoder = LabelEncoder()
#train.iloc[:,6] = labelencoder.fit_transform(train.iloc[:,6])
train.iloc[:,1] = labelencoder.fit_transform(train.iloc[:,1])
onehot = OneHotEncoder(categorical_features = [0,1,3])
train_x = onehot.fit_transform(train.values).toarray()

one_hot = OneHotEncoder(categorical_features = [0]) 
train_y = one_hot.fit_transform(train_y.values).toarray()
#model parameters
total_size, total_features =train_x.shape
batch_size = 20
hidden_layers =50
input_size = 14
output_size = 2
iterations =2000

## next batch
def getnextbatch():
    i = int(random.choice(np.arange(0,total_size - batch_size)))
    x_batch = train_x[i:i+batch_size]
    y_batch = train_y[i:i+batch_size]
    return x_batch, y_batch

## input and output
x_input = tf.placeholder(shape = [batch_size,input_size], dtype = tf.float64, name = "x_input")
y_input = tf.placeholder(shape = [batch_size, output_size], dtype = tf.float64, name = "y_input")

## z = weights * x_input + y_input
#layer1 = input_layer*wi + bi
#layer1 = relu(layer1)
#layer 2 = layer1*w2+b2
#..
#output  = 

weight = {
        'h1' : tf.Variable(np.zeros(shape = [input_size,hidden_layers]), name = "h1"),
        'h2' : tf.Variable(np.zeros(shape = [hidden_layers,hidden_layers]), name = "h2"),
        'h3' : tf.Variable(np.zeros(shape = [hidden_layers,output_size]), name = "h3")
        }

bias = {
        'b1' : tf.Variable(np.zeros(shape = [hidden_layers]), name = "b1"),
        'b2' : tf.Variable(np.zeros(shape = [hidden_layers]), name = "b2"),
        'b3' : tf.Variable(np.zeros(shape = [output_size]), name = "b3")
        }

layer1 = tf.add(tf.matmul(x_input,weight['h1']), bias['b1'], name = "layer1")
layer1 = tf.nn.relu(layer1, name = "relu1")
layer2 = tf.add(tf.matmul(layer1,weight['h2']),bias['b2'], name = "layer2")
layer2 = tf.nn.relu(layer2 , name = "relu2")
output = tf.add(tf.matmul(layer2,weight['h3']), bias['b3'], name = "output")
output = tf.nn.softmax(output, name = "softmax")

loss =tf.reduce_mean(tf.square(output - y_input))

logdir = "tf_logs/current1"
loss_summary = tf.summary.scalar('mse',loss)
fileWriter = tf.summary.FileWriter(logdir)
#ls -l tf-logs/run*  list the contents
#tensorboard --logdir tf_logs/

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001, name = "adamOptimizer")
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    x_i,y_i = getnextbatch()
    for i in np.arange(0,iterations):
        sess.run(train,feed_dict = {x_input:x_i,y_input:y_i})
        if i%10 == 0:
            mse = loss_summary.eval(feed_dict ={x_input: x_i, y_input: y_i})
            print(mse)
            fileWriter.add_summary(mse,i)
            print(i,"\tmse",mse)
        saver.save(sess,"./machine_2/mario")
  

for i in np.where(np.isnan(test.iloc[:,4])):
    test.iloc[i,4] = 0

#test['Embarked'] = test['Embarked'].map(lambda x : 0 if x == 'S' else 1 if x =='C' else 2 if x == 'Q' else 0)
labelencoder = LabelEncoder()
#test.iloc[:,6] = labelencoder.fit_transform(test.iloc[:,6])
test.iloc[:,1] = labelencoder.fit_transform(test.iloc[:,1])
onehot = OneHotEncoder(categorical_features = [0,1,3])
test_x = onehot.fit_transform(test.values).toarray() 
      
with tf.Session() as sess:
    saver.restore(sess,"./machine_2/mario")
    X_new = train_x[0:20]
    y_pred = sess.run(output,feed_dict = {x_input:X_new})

y_pred_ = []
for i,y in enumerate(y_pred):
    print(i)
    y_pred_[i] = 0 if y[0]> y[1] else 1
fileWriter.close()




























