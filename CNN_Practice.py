import numpy as np 
import pandas as pd 
import keras
import pickle
import tensorflow as tf
#import keras.utils as np_utils
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

#Load Dataseta
#(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()
#print(xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)

import tensorflow.keras as tk
mnist = tk.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

#Plot1
# fig = plt.gcf()
# fig.set_size_inches(2, 2)
# plt.imshow(xTrain[0], cmap='binary')
# plt.show()
# print(yTrain[0])

#plot2
# fig = plt.gcf()
# fig.set_size_inches(12, 14)
# num=15
# for i in range(0, num):
#     ax=plt.subplot(5,5, 1+i)
#     ax.imshow(xTrain[i], cmap='binary')
#     title= "label=" +str(yTrain[i])
#     ax.set_title(title,fontsize=10) 
#     ax.set_xticks([]);ax.set_yticks([])        
#     i+=1 
# plt.show()

#Add 1 dimension
xTrain4D=xTrain.reshape(xTrain.shape[0],28,28,1).astype('float32')
xTest4D=xTest.reshape(xTest.shape[0],28,28,1).astype('float32')
#print(xTrain4D.shape, xTest4D.shape)

#Normalize
#xTrain4D = xTrain4D / 255
xTrain4D = xTrain4D[50000:51000,:] / 255
xTest4D = xTest4D / 255

#Onehot encoding
#yTrain = np_utils.to_categorical(yTrain) # method 1
yTrain = keras.utils.np_utils.to_categorical(yTrain) # method 2
#yTrain = keras.utils.to_categorical(yTrain) # method 3
yTrain = yTrain[50000:51000,:]
#yTest = np_utils.to_categorical(yTest) # method 1
yTest = keras.utils.np_utils.to_categorical(yTest) # method 2
#yTest = keras.utils.to_categorical(yTest) # method 3


#Set CNN Model
model = Sequential()
#Convolution
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1), 
                 activation='relu'))
#Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#Convolution
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
#Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#Drop out to avoid overfitting
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
#print(model.summary())

#Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
train_history=model.fit(x=xTrain4D, y=yTrain, validation_split=0.2,
                        epochs=20, batch_size=300,verbose=2)

#print the training history
plt.plot(train_history.history['accuracy']) #'accuracy', 'loss'
plt.plot(train_history.history['val_accuracy']) #'val_accuracy', 'val_loss'
plt.title('Train History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Save the Model
# filename = 'CNN.sav'
# pickle.dump(model, open(filename, 'wb'))
filename = 'CNN'
tf.keras.models.save_model(model, filename)
print(model.summary())
