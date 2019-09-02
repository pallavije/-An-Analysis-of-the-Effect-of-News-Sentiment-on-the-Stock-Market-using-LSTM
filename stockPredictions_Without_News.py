# Importing the libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


# loading dataset

dataset = pd.read_csv("AppleNewsStock.csv")

# dropping news column

dataset.drop(columns="News", inplace=True)
print(dataset.head())

# visualizing the dataset

plt.figure()
plt.plot(dataset["Open"])
plt.plot(dataset["High"])
plt.plot(dataset["Low"])
plt.plot(dataset["Close"])
plt.title('Apple stock price history')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open', 'High', 'Low', 'Close'])
plt.show()

# Plotting the volume history

plt.figure()
plt.plot(dataset["Volume"])
plt.title('Apple stock price history')
plt.ylabel('Volume')
plt.xlabel('Days')
plt.show()

# checking null values in the dataset

print(dataset.isna().sum())


# splitting dataset into train and test data

features = ["Open","High","Low","Close",'Adj Close', 'Volume', 'Index']
train, test = train_test_split(dataset, test_size=0.2, random_state=3)

print("Train data shape : ", train.shape)
print("Test data shape : ", test.shape)

print("Sample Input data")
print(train.iloc[0:2])

# normalization

scaler = MinMaxScaler()
normalizedTrain = scaler.fit_transform(train[features])
normalizedTest = scaler.fit_transform(test[features])

print("Post normalization - shape of data :-")
print("\tNormalized train data shape : ", normalizedTrain.shape)
print("\tNormalized test data shape : ", normalizedTest.shape)
print("\tSample Input data")
print(normalizedTrain[0:2])

# converting data into 3D data for LSTM in the timestep of 2 months

X_train = []
y_train = []
for i in range(60, len(normalizedTrain)):
    X_train.append(normalizedTrain[i-60:i, 0])
    y_train.append(normalizedTrain[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print("3D data shape both input and output")
print(X_train.shape)
print(y_train.shape)

# defining model

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Training

trainedModel = model.fit(X_train, y_train, epochs=100, batch_size=32)

# plotting the results

plt.figure(figsize=(16, 5))
plt.plot(trainedModel.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()

X_test = []
for i in range(60, len(normalizedTest)):
    X_test.append(normalizedTest[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted = model.predict(X_test)
predictedDataset = np.zeros(shape=(len(predicted), 7))
predictedDataset[:, 0] = predicted[:, 0]
predict = scaler.inverse_transform(predictedDataset)[:, 0]
print(predict)


# Visualising the results

plt.plot(predict, color='blue', label='Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()
