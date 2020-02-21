# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:45:55 2020

@author: nive4
"""
# Importing the dataset from Keras 

from keras.datasets import fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
c_test_labels=test_labels
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np


# confusion Matrix
print("Confusion Matrix : ")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd



# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
output_1 = Dense(512, activation='relu')(inputs)
predictions = Dense(10, activation='softmax')(output_1)

#Reshaping the data

train_images = train_images.reshape(60000,28*28)
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255

# Convert labels to categories
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# This creates a model that includes
# the Input layer and  Dense layers
model = Model(inputs=inputs, outputs=predictions)

#Compiling the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

#fitting the model 
model.fit(train_images, train_labels, epochs=5, batch_size=128)

#predicting the model 
pred=model.predict(test_images)
list_pred = list(map(np.argmax,pred))
test_loss, test_acc=model.evaluate(test_images, test_labels)
conf_matx = confusion_matrix(c_test_labels,list_pred)

#print(pd.DataFrame(conf_matx, index=names, columns=names))
print(conf_matx)
model.fit(train_images, train_labels, epochs=20, batch_size=128)
print('test_acc:',test_acc)
