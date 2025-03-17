# -*- coding: utf-8 -*-
#Practical 9
#Predicting House prices:A regression example

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout 
import matplotlib.pyplot as plt

data=fetch_california_housing()

#Splitting into x & y
x=data.data
y=data.target
#split dataset into train,test,validation
x_train,x_temp,y_train,y_temp=train_test_split(x,y,test_size=0.3,random_state=42)
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.5,random_state=42)
#standardise the dataset
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)


print(x_train.shape[1])
#This is the input shape
#Build the model
model=Sequential(
    [
     Dense(64,activation='relu',input_shape=(x_train.shape[1],)), 
     Dropout(0.2),
     Dense(32,activation='relu'),
     Dropout(0.2),
     Dense(1)])
#Compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae'])
#print summary
model.summary()

#Train the model
history=model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=20,batch_size=32,verbose=1)#model takes 32 samples in a single iteration


#Evaluate the model
test_loss,test_mae=model.evaluate(x_test,y_test)
print("Test loss",test_loss)
print("Test MAE",test_mae)

print(history.history)

#Plotting training and validation loss
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['mae'],label='Training MAE')
plt.plot(history.history['val_mae'],label='Validation MAE')
plt.xlabel("Epoch")
plt.ylabel('MAE')
plt.legend()
plt.show()

#If number of layers increased
model=Sequential([
    Dense(64,activation='relu',input_shape=(x_train.shape[1],)),
    Dropout(0.2),
    Dense(32,activation='relu'),
    Dropout(0.2),
    Dense(16,activation='relu'),
    Dense(1)])

#Compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae'])
#print summary
model.summary()

#Train the model
history=model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=20,batch_size=32,verbose=1)#model takes 32 samples in a single iteration

#Evaluate the model
test_loss,test_mae=model.evaluate(x_test,y_test)
print("Test loss",test_loss)
print("Test MAE",test_mae)

print(history.history)

#Plotting training and validation loss
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['mae'],label='Training MAE')
plt.plot(history.history['val_mae'],label='Validation MAE')
plt.xlabel("Epoch")
plt.ylabel('MAE')
plt.legend()
plt.show()



