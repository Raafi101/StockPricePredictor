#Stock Price Predictor

import tensorflow as tf
from tensorflow import keras
from keras.layers.recurrent import LSTM
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import math
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
import time

#Get stock prices, df = dataframe (year-month-day)
df = web.DataReader('TSLA', data_source='yahoo', start= '2000-01-01', end='2020-10-01')
#print data
print(df)

#Get the shape of the data
print(df.shape)

#Make a matplotlib chart of price
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price in $', fontsize=18)

#Create new dataframe with only close price
data = df.filter(['Close']) #Data is the new df
#Make df into numpy array
dataset = data.values

#scale data. Make the max input 1 and the min 0 and transform all else
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Print scaled data
print(scaled_data)

#Divide data
train_data_index = math.ceil(len(dataset)*0.8)

train_data = scaled_data[0:train_data_index, :]

x_train = [] #Will store past 60 prices
y_train = [] #Next price we want to figure out
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0 ])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()

#Convert x and y train to numpy arrays

x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape data. We have 2 dimensions and lstm expects 3 dimensions
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build LSTM model. LSTM is similar to RNN i think
model = keras.Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(keras.layers.Dense(25))
model.add(keras.layers.Dense(1))

#Compile
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#Train
model.fit(x_train, y_train, batch_size=3, epochs=5)

#Create testing data
test_data = scaled_data[train_data_index - 60: , :]
x_test = []
y_test = dataset[train_data_index: , :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0 ])

#Convert data to numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get models predicted price
prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)

#Use root mean squared error to see how accurate the model is
rmse = np.sqrt(np.mean(((prediction - y_test)**2)))
print(rmse)

#Plot data
train = data[:train_data_index]
valid = data[train_data_index:]
valid['Prediction'] = prediction
plt.figure(figsize=(16,8))
plt.title('Tesla Price Predictor')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price in USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Prediction']])
plt.legend(['Train', 'Validation', 'Prediction'], loc='lower right')
plt.show()

#Compare predict and actual prices
print(valid)

#Predict future prices
quote = web.DataReader('TSLA', data_source='yahoo', start= '2000-01-01', end='2020-10-01')
new_df = quote.filter(['Close'])

#Get last 60 days and put into array
last_60_days = new_df[-60:].values

#scale
last_60_days_scaled = scaler.transform(last_60_days)

#Create empty list
x2_test = []
x2_test.append(last_60_days_scaled)

#Convert x2_test data to numpy array
x2_test = np.array(x2_test)

#Reshape
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1], 1))

#Get scaled price
predicted_price = model.predict(x2_test)

#Undo scaling
predicted_price = scaler.inverse_transform(predicted_price)

#predicted price
results = model.evaluate(x_test, y_test)
print("Predicted Price: ", predicted_price)