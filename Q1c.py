# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:23:46 2020

@author: nive4
"""

# Importing the dataset from Keras 

from keras.datasets import fashion_mnist
print("Loading Fashion MNIST Data ...")
(train_images, train_labels) , (test_images, test_labels) =  fashion_mnist.load_data()
c_test_labels = test_labels
names=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print("Fashion MNIST data loaded successfully.")


# Reshape the data
print("Reshaping data ...")
train_images = train_images.reshape(60000,28*28)
train_images = train_images.astype('float32')/255
test_images = test_images.reshape(10000,28*28)
test_images = test_images.astype('float32')/255

# Convert labels to categories
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("Labels converted to categorical ... ")

from keras import models
from keras import layers

#Defining the model type 
print("Building One-Hidden layer Seqeuntial Neural Network ... ")
network = models.Sequential()
network.compile( optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Adding layers to the model
network.add(layers.Dense(512, activation='relu' , input_shape=(28*28,) ) )
network.add(layers.Dense(10, activation='softmax'))

# Fit the model
print("Fiting the Sequential model with Train Data ... ")
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluate the model
print("Evaluating the Sequential model with Test Data ...")
test_loss, test_accuracy = network.evaluate(test_images, test_labels)

print('Test Accuracy : ', test_accuracy) #Test Accuracy :  0.8695999979972839
print('Test Loss : ', test_loss) # Test Loss :  0.36936546564102174


# Predicting
pred = network.predict(test_images)

# confusion Matrix
print("Confusion Matrix : ")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd




list_pred = list(map(np.argmax,pred))


# confusion Matrix
conf_matx = confusion_matrix(c_test_labels,list_pred)
#print(pd.DataFrame(conf_matx, index=names, columns=names))
print(conf_matx)


#Visualization
print("Visualizing Confusion Matrix as Heatmap : plot visible in the Plots tab")
import seaborn as sns
sns.heatmap(conf_matx, cmap ="Spectral", xticklabels=names , yticklabels=names )
plt.show()

# setting  counters for Misclassification and Correct Second Predictions
misclasscount = 0
secondpred = 0

#printing probabilities

for i in range (len(test_images)):
    if c_test_labels[i] != np.argmax(pred[i]):
        misclasscount = misclasscount+1
        plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
        plt.show()
        print("Predicted value of ",i,': ',names[np.argmax(pred[i])], " ;  Actual value : ",names[c_test_labels[i]])
        #print(pred[i])
        plt.figure(figsize = (15,10))
        plt.bar(height=pred[i], x=names, width=0.2)
        plt.show()
        # Predicting second highest
        df_prob = pd.DataFrame(names, pred[i])
        df_prob = df_prob.reset_index()
        df_prob.columns = ['Prob','Names']
        df_prob=df_prob.sort_values(by=['Prob'], ascending=False).reset_index()
        print(df_prob)
        print('Second Highest Probability for : ',df_prob.loc[1,'Names'],' ; Prob = ',df_prob.loc[1,'Prob'])
        # Count the number of correct second highest predictions
        if names[c_test_labels[17]] == df_prob.loc[1,'Names'] :
            secondpred = secondpred +1
        # Continue ?
        if input("Continue : (y/n) ") == 'n':
            break

#print('Misclasscount : ',misclasscount, ' ; Second Pred count : ',secondpred)
misclasscount = 0
secondpred = 0

for i in range(len(test_images)):
    if c_test_labels[i] != np.argmax(pred[i]):
        misclasscount = misclasscount +1
        # Predicting second highest
        df_prob = pd.DataFrame(names, pred[i])
        df_prob = df_prob.reset_index()
        df_prob.columns = ['Prob','Names']
        df_prob=df_prob.sort_values(by=['Prob'], ascending=False).reset_index()
        #print('Second Highest Probability for : ',df_prob.loc[1,'Names'],' ; Prob = ',df_prob.loc[1,'Prob'])
        # Count the number of correct second highest predictions
        if names[c_test_labels[i]] == df_prob.loc[1,'Names']:
            secondpred = secondpred +1
        # Continue ?

print('Total misclassifications in Test Data : ',misclasscount)
print('Total misclassifications where second prediction was correct : ', secondpred)
print("Total misclass percentage : ",(misclasscount/len(test_labels))*100)
print("Percentage where second pred was correct : ",(secondpred/len(test_labels))*100)
print("Number of second pred was correct :", secondpred)


# Fit the model
print("Fiting the Sequential model with Train Data ... ")
network.fit(train_images, train_labels, epochs=20, batch_size=128)

# Evaluate the model
print("Evaluating the Sequential model with Test Data ...")
test_loss, test_accuracy = network.evaluate(test_images, test_labels)

print('Test Accuracy : ', test_accuracy) #Test Accuracy :  0.8820000290870667
print('Test Loss : ', test_loss) #Test Loss :  0.44568557067513465


# Predicting
pred = network.predict(test_images)

# confusion Matrix
print("Confusion Matrix : ")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd




list_pred = list(map(np.argmax,pred))


    
conf_matx = confusion_matrix(c_test_labels,list_pred)
#print(pd.DataFrame(conf_matx, index=names, columns=names))
print(conf_matx)


#Visualization
print("Visualizing Confusion Matrix as Heatmap : plot visible in the Plots tab")
import seaborn as sns
sns.heatmap(conf_matx, cmap ="Spectral", xticklabels=names , yticklabels=names )
plt.show()

# Set counters for Misclassification and Correct Second Predictions
misclasscount = 0
secondpred = 0

#printing probabilities

for i in range (len(test_images)):
    if c_test_labels[i] != np.argmax(pred[i]):
        misclasscount = misclasscount+1
        plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
        plt.show()
        print("Predicted value of ",i,': ',names[np.argmax(pred[i])], " ;  Actual value : ",names[c_test_labels[i]])
        #print(pred[i])
        plt.figure(figsize = (15,10))
        plt.bar(height=pred[i], x=names, width=0.2)
        plt.show()
        # Predicting second highest
        df_prob = pd.DataFrame(names, pred[i])
        df_prob = df_prob.reset_index()
        df_prob.columns = ['Prob','Names']
        df_prob=df_prob.sort_values(by=['Prob'], ascending=False).reset_index()
        print(df_prob)
        print('Second Highest Probability for : ',df_prob.loc[1,'Names'],' ; Prob = ',df_prob.loc[1,'Prob'])
        # Count the number of correct second highest predictions
        if names[c_test_labels[17]] == df_prob.loc[1,'Names'] :
            secondpred = secondpred +1
        # Continue ?
        if input("Continue : (y/n) ") == 'n':
            break

#print('Misclasscount : ',misclasscount, ' ; Second Pred count : ',secondpred)
misclasscount = 0
secondpred = 0

for i in range(len(test_images)):
    if c_test_labels[i] != np.argmax(pred[i]):
        misclasscount = misclasscount +1
        # Predicting second highest
        df_prob = pd.DataFrame(names, pred[i])
        df_prob = df_prob.reset_index()
        df_prob.columns = ['Prob','Names']
        df_prob=df_prob.sort_values(by=['Prob'], ascending=False).reset_index()
        
        if names[c_test_labels[i]] == df_prob.loc[1,'Names']:
            secondpred = secondpred +1
        # Continue ?

print('Total misclassifications in Test Data : ',misclasscount)
print('Total misclassifications where second prediction was correct : ', secondpred)
print("Total misclass percentage : ",(misclasscount/len(test_labels))*100)
print("Percentage where second pred was correct : ",(secondpred/len(test_labels))*100)
print("Number of second pred was correct :", secondpred)