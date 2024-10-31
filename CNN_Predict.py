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

#Load the Model
# filename = 'CNN.sav'
# model = pickle.load(open(filename, 'rb'))
filename = 'CNN'
model = tf.keras.models.load_model(filename)

#Load Dataseta
(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()

#Add 1 dimension
xTest4D=xTest.reshape(xTest.shape[0],28,28,1).astype('float32')
#Normalize
xTest4D = xTest4D[5000:5100,:] / 255
#Onehot encoding
#yTest1hot = np_utils.to_categorical(yTest)
yTest1hot = keras.utils.np_utils.to_categorical(yTest)
yTest1hot = yTest1hot[5000:5100,:]

#Evaluate
scores = model.evaluate(xTest4D , yTest1hot)
print('Score is ' + str(round((scores[1]*100), 2)) + ' %')

# #Prediction
# prediction = model.predict_classes(xTest4D)
# #print(prediction[:10])

# #Confusion Matrix
# print(pd.crosstab(yTest[5000:5100], prediction,
#             rownames=['label'], colnames=['predict']))
