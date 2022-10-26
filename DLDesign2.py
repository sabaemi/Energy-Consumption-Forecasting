#Prefered model

#Deep learning network design, with normalization, removal of outlier data, initial model
"""
Libraries
"""

# Data related
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Deep learning
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Plot
import matplotlib.pyplot as plt
# %matplotlib inline

"""Reading Dataset"""

#train
# the data needs to be in train.xlsx and test.xlsx files

read_file = pd.read_excel('train.xlsx')
read_file.to_csv ("train.csv", index = None, header=True)
train = pd.DataFrame(pd.read_csv("train.csv"))

#combine columns and changing their name
train.columns = ['Date', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00', '00:00']
train = train.set_index(["Date"]).stack().reset_index().rename(columns={"level_1": "new"})
train['Date']=train['Date'] +' '+ train['new']

train.drop('new', inplace=True, axis=1)
train.columns = ['Datetime', 'MWH']
train

"""Visualization of the time series"""

# Ploting train sample
plt.figure(figsize=(12, 8))
# plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
plt.plot(train.index ,train['MWH'])
# plt.plot(train)
plt.xlabel("Date")
plt.ylabel("MegaWatts")
plt.title("plot for the energy consumption time series")
plt.show()

"""Visualization based on year"""

# Ploting train sample
plt.figure(figsize=(25, 8))
plt.style.use('ggplot')
plt.plot(train['Datetime'] ,train['MWH'])
plt.xlim(0,70128)
plt.ylim(0,70000)
plt.xticks(np.arange(0, 70128, step=720))
plt.xticks(rotation=90)
plt.xlabel("Date")
plt.ylabel("MegaWatts")
plt.title("plot for the energy consumption time series")
plt.show()

"""histogram for the energy consumption in trainingset"""

#histogram
plt.figure(figsize=(20, 8))
plt.xlabel("MegaWatts")
plt.ylabel("Count")
plt.title("plot histogram for the energy consumption ")
plt.hist(train['MWH'])
plt.show()

#test

read_file = pd.read_excel('test.xlsx')
read_file.to_csv ("test.csv", index = None, header=True)
test = pd.DataFrame(pd.read_csv("test.csv"))

#combine columns and changing their name
test.columns = ['Date', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00', '00:00']
test = test.set_index(["Date"]).stack().reset_index().rename(columns={"level_1": "new"})
test['Date']=test['Date'] +' '+ test['new']

test.drop('new', inplace=True, axis=1)
test.columns = ['Datetime', 'MWH']
test

"""Visualization of the time series

"""

# Ploting test sample
plt.figure(figsize=(12, 8))
# plt.style.use('fivethirtyeight')
plt.plot(test.index, test['MWH'])
# plt.plot(test)
plt.xlabel("Date")
plt.ylabel("MegaWatts")
plt.title("plot for the energy consumption time series")
plt.show()

#year
# Ploting test sample
plt.figure(figsize=(20, 8))
# plt.style.use('fivethirtyeight')
plt.plot(test['Datetime'], test['MWH'])
plt.xlim(0,26304)
plt.ylim(0,70000)
plt.xticks(np.arange(0, 26304, step=720))
plt.xticks(rotation=90)
# plt.plot(test)
plt.xlabel("Date")
plt.ylabel("MegaWatts")
plt.title("plot for the energy consumption time series")
plt.show()

#histogram
plt.figure(figsize=(20, 8))
plt.xlabel("MegaWatts")
plt.ylabel("Count")
plt.title("plot histogram for the energy consumption ")
plt.hist(test['MWH'])
plt.show()

"""Preprocessing"""

#nulls
print("number of nulls in trainset is " +str(train.isna().sum().sum()))
print("number of nulls in testset is " +str(test.isna().sum().sum()))

#zeros
# Count number of rows matching a condition
row_numbers = train[(train['MWH'] == 0)].index
if(len(row_numbers)!=0):
  print("number of zeros in trainset is " +str(len(row_numbers)))
  print(row_numbers)

row_numbers = test[(test['MWH'] == 0)].index
if(len(row_numbers)!=0):
  print("number of zeros in testset is " +str(len(row_numbers)))
  print(row_numbers)

#maxvalues
column = train['MWH']
max_value = column.max()
max_index = column.idxmax()
print("max value in trainset is " +str(max_value))
print("max index in trainset is " +str(max_index))

#replacing zeros with means
row_numbers = train[(train['MWH'] == 0)].index
for i in row_numbers:
  train['MWH'][i]=(train['MWH'][i-1]+train['MWH'][i+1])/2

row_numbers = test[(test['MWH'] == 0)].index
for i in row_numbers:
  test['MWH'][i]=(test['MWH'][i-1]+test['MWH'][i+1])/2

"""with normalization"""

# fix random seed for reproducibility
np.random.seed(7)
# normalize the dataset
train.drop('Datetime', inplace=True, axis=1)
train = train.values
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
train

# normalize the dataset
test.drop('Datetime', inplace=True, axis=1)
test = test.values
scaler = MinMaxScaler(feature_range=(0, 1))
test = scaler.fit_transform(test)
test

"""without normalization"""

# #train
# train.drop('Datetime', inplace=True, axis=1)
# train = train.values

# #test
# test.drop('Datetime', inplace=True, axis=1)
# test = test.values

"""create x and y"""

# create X and Y matrix from time series for training
def create_X_Y(dataset: list, look_back: int):
    X, Y = [], []
    if len(dataset) - look_back <= 0:
        X.append(dataset)
    else:
        for i in range(len(dataset) - look_back):
            Y.append(dataset[i + look_back])
            X.append(dataset[i:(i + look_back)])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1],1))  # for time step

    return X, Y

look_back = 3
trainX, trainY = create_X_Y(train, look_back)
print(trainX, trainY)

testX, testY = create_X_Y(test, look_back)
print(testX, testY)

"""model"""

# model the LSTM network
model = Sequential()
# model.add(LSTM(4,activation='relu', input_shape=(1,look_back)))   # for without normalization
model.add(LSTM(4, input_shape=(look_back,1)))   # for time step
# model.add(LSTM(4, input_shape=(1,look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, validation_split = 0.1,epochs=20, batch_size=64, verbose=2)
# act func sigmoid
# 20 epoch
# 64 bachsize

# predictions
trainPredict = model.predict(trainX)
print(trainPredict)
testPredict = model.predict(testX)
print(trainPredict)

"""error"""

# calculate test and train score

# normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

"""plot"""

# shift train predictions for plotting
trainPredictPlot = np.empty_like(train)
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict

# plt.style.use('fivethirtyeight')
# Ploting train sample
plt.figure(figsize=(12, 8))
# plt.plot(train.index ,train['MWH'])
plt.plot(scaler.inverse_transform(train))
plt.plot(trainPredictPlot)
plt.xlabel("Date")
plt.ylabel("MegaWatts")
plt.legend(['train', 'predict'], loc='upper left')
plt.show()

# shift test predictions for plotting
testPredictPlot = np.empty_like(test)
testPredictPlot[:] = np.nan
testPredictPlot[look_back:len(testPredict)+look_back] = testPredict

# Ploting train sample
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(test))
plt.plot(testPredictPlot)
plt.xlabel("Date")
plt.ylabel("MegaWatts")
plt.legend(['test', 'predict'], loc='upper left')
plt.show()

"""model loss"""

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend('train', loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend( 'val', loc='upper left')

plt.show()

"""combine train and test"""

#data
# all of the data needs to be data.xlsx file
read_file = pd.read_excel('data.xlsx')
read_file.to_csv ("data.csv", index = None, header=True)
data = pd.DataFrame(pd.read_csv("data.csv"))

#combine columns and changing their name
data.columns = ['Date', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00', '00:00']
data = data.set_index(["Date"]).stack().reset_index().rename(columns={"level_1": "new"})
data['Date']=data['Date'] +' '+ data['new']

data.drop('new', inplace=True, axis=1)
data.columns = ['Datetime', 'MWH']
data

"""Visualization of the time series"""

# Ploting test sample
plt.figure(figsize=(12, 8))
# plt.style.use('fivethirtyeight')
plt.plot(data.index, data['MWH'])
# plt.plot(test)
plt.xlabel("Date")
plt.ylabel("MegaWatts")
plt.title("plot for the energy consumption time series")
plt.show()

row_numbers = data[(data['MWH'] == 0)].index
for i in row_numbers:
  data['MWH'][i]=(data['MWH'][i-1]+data['MWH'][i+1])/2

# normalize the dataset
data.drop('Datetime', inplace=True, axis=1)
data = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
data

"""predicting future"""

look_back = 24
trainX, trainY = create_X_Y(train, look_back)
testX, testY = create_X_Y(test, look_back)

# Forecasting future 24 hours
y = data
X,Y = create_X_Y(y, look_back)

# model the LSTM network
model = Sequential()
# model.add(LSTM(4,activation='relu', input_shape=(1,look_back)))
model.add(LSTM(4, input_shape=(look_back,1)))   # for time step
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X, Y,epochs=20, batch_size=64, verbose=2)
# act func sigmoid
# 20 epoch
# 64 bachsize

#prediction
y = y[-look_back:]
predictlist = []
X,_ = create_X_Y(y, look_back)
n_ahead = 24
for i in range(n_ahead):
  # Making the prediction
  future = model.predict(X)
  predictlist.append(future)
  # input matrix for forecasting first apend with privious data then reshape for next iteration
  X = np.append(X, future)
  X = np.delete(X, 0)
  X = np.reshape(X, (1, len(X), 1))

predictlist = [[y[0][0]] for y in predictlist]
predictlist = scaler.inverse_transform(predictlist)
print(predictlist)

#plotting future
plt.figure(figsize=(12, 8))
plt.plot(predictlist)
plt.xlabel("Hour")
plt.ylabel("MegaWatts")
plt.legend(['future'], loc='upper left')
plt.show()