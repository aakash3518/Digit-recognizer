# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 22:16:04 2018

@author: Aakash
"""

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score

# Encoding labels for categorical data
def labelencoder(label):
    l = np.zeros((label.shape[0],10))
    for i in range(len(label)):
        l[i,label[i]] = 1
    return l

# Decoding labels for categorical data
def labeldecoder(label):
    l = np.zeros((label.shape[0],1))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j] == 1:
                l[i,0] = j
    return l

# Importing training and testsets
dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')

y = dataset.iloc[:, 0:1].values
X = dataset.iloc[:, 1:].values

Xt = testset.iloc[:, :].values

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X)
y_train = labelencoder(y)

sct = StandardScaler()
X_test = sct.fit_transform(Xt)

# Importing Keras libraries
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 300, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dense(units = 300, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 80, epochs = 100)

# Predicting the testset labels
y_pred = classifier.predict(X_test)

# Optimising the predictions to have only 0's and 1's
for i in range(y_pred.shape[0]):
    for j in range(y_pred.shape[1]):
        if y_pred[i,j] != 1:
            y_pred[i,j] = 0

# Decoding predictions
predictions = labeldecoder(y_pred)

tmp = np.concatenate((np.arange(1,28001).reshape(28000,1),predictions),axis = 1)
# Data frame of predictions
df = pd.DataFrame(tmp,columns = ['ImageId','Label'])
# Saving predictions to csv
df.to_csv('predictions.csv')
# printing testset accuracy
#print('Accuracy:',accuracy_score(y_test,y_pred))