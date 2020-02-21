# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:43:06 2020

@author: nive4
"""

# Importing the dataset from Keras 


from keras.datasets import fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

from keras.layers import Input, Dense, Flatten
from keras.models import Model
import numpy as np
c_test_labels=test_labels

inputs = Input(shape=(28,28))
temp = Flatten()(inputs)
intermediate_out = Dense(512, activation='sigmoid')(temp)
output=(Dense(10,activation='softmax'))(intermediate_out)

# confusion Matrix
print("Confusion Matrix : ")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


model = Model(inputs=inputs, outputs=output)
#Compiling the model
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
train_images=train_images.astype('float32')/255
test_images=test_images.astype('float32')/255


from keras.utils import to_categorical

# Convert labels to categories
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Fitting the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc=model.evaluate(test_images, test_labels)
pred=model.predict(test_images)
list_pred = list(map(np.argmax,pred))

# confusion Matrix

conf_matx = confusion_matrix(c_test_labels,list_pred)
print(conf_matx)

#Fitting the model for epoch =20
model.fit(train_images, train_labels, epochs=20, batch_size=128)
model.summary()