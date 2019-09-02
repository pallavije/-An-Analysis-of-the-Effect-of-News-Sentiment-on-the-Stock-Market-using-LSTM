
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential



# loading two different datasets

df = pd.read_csv("./AppleNewsStock.csv")

# considering apple stock data as primary data

dataset = df

#creating dataframe with date and the target variable

stockData=dataset["Open"]
dataset=dataset.drop(['Date','News','Adj Close','High','Low'],axis=1)  

#scaling data

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(dataset)
print(stockData.head())
print(dataset.head())

print(stockData.shape)

# checking for any value

print("Stock data result : ", stockData.isnull().any(), sep="\n")

# time to normalize opening price

minCost = min(stockData)
maxCost = max(stockData)
misc = dict()
misc['minCost'] = minCost
misc['maxCost'] = maxCost
np.save("./misc.npy",misc)

def normalize(cost):
    return (cost - minCost) / (maxCost - minCost)


normalizedstockData = list(map(normalize, stockData))

print()
print('Before normalization :- ')
print(minCost)
print(maxCost)
print(np.mean(stockData))

print()
print('After normalization :- ')
print(min(normalizedstockData))
print(max(normalizedstockData))
print(np.mean(normalizedstockData))


normalized = np.asarray(normalizedstockData)
print(normalized)
normalized = pd.DataFrame(data=normalized.flatten())

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(normalized)

# splitting the dataset into train 

X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Building the LSTM

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 50, batch_size = 62)


# Predicting close using the Test Set

dataset_total = pd.concat((dataset['Open'], normalized), axis = 0)
inputs = dataset_total[len(dataset_total) - len(normalized) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


X_test = []
y_test= []
for i in range(60, 1500):
    X_test.append(inputs[i-60:i, 0])
    y_test.append(inputs[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)


X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)

error = mse(y_test, predicted_stock_price)

print()
print("Mean Square Error : ", error)

def reverse_normalization(cost):
    return cost * (maxCost - minCost) + minCost


reverNormalized_prediction = list(map(reverse_normalization, predicted_stock_price))
reverNormalized_Ytest = list(map(reverse_normalization, y_test))

#Plotting the Results

plt.plot(reverNormalized_prediction, color = 'black', label = 'Apple Stock Price')
plt.plot(reverNormalized_Ytest, color = 'green', label = 'Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Instances')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()
