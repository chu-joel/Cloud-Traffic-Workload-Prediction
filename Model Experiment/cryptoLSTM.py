import ccxt
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow import keras
from keras.layers import Dense

print("loading data")
ex = ccxt.binance()

# download data from binance spot market
df = pd.DataFrame(
    ex.fetch_ohlcv(symbol='BTCUSDT', timeframe='1d', limit=1000),
    columns=['unix', 'open', 'high', 'low', 'close', 'volume']
)


# convert unix (in milliseconds) to UTC time
df['date'] = pd.to_datetime(df.unix, unit='ms')


# Formatting the grids for display
rcParams['figure.figsize'] = 14, 8
sns.set(style='whitegrid', palette='muted', font_scale=1.5)


scaler = MinMaxScaler()
# fit the format of the scaler -> convert shape from (1000, ) -> (1000, 1)
close_price = df.close.values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_price)


seq_len = 60


def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

# Split data into training and test data


def get_train_test_sets(data, seq_len, train_frac):
    sequences = split_into_sequences(data, seq_len)
    n_train = int(sequences.shape[0] * train_frac)
    x_train = sequences[:n_train, :-1, :]
    y_train = sequences[:n_train, -1, :]
    x_test = sequences[n_train:, :-1, :]
    y_test = sequences[n_train:, -1, :]
    return x_train, y_train, x_test, y_test


print("preprocessing data")
x_train, y_train, x_test, y_test = get_train_test_sets(
    scaled_close, seq_len, train_frac=0.9)

print(x_train.size)
print(y_train.size)
print(x_test.size)
print(y_test.size)
print("Building LSTM model")
# fraction of the input to drop; helps prevent overfitting
dropout = 0.2
window_size = seq_len - 1

# build a 3-layer LSTM RNN
model = keras.Sequential()

model.add(
    LSTM(window_size, return_sequences=True,
         input_shape=(window_size, x_train.shape[-1]))
)

model.add(Dropout(rate=dropout))
# Bidirectional allows for training of sequence data forwards and backwards
model.add(
    Bidirectional(LSTM((window_size * 2), return_sequences=True)
                  ))

model.add(Dropout(rate=dropout))
model.add(
    Bidirectional(LSTM(
        window_size, return_sequences=False))
)

model.add(Dense(units=1))
# linear activation function: activation is proportional to the input
model.add(Activation('linear'))


print("training and evaluating")
batch_size = 16

model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

history = model.fit(
    x_train,
    y_train,
    epochs=1,
    batch_size=batch_size,
    shuffle=False,
    validation_split=0.2
)

print("Making prediction")
y_pred = model.predict(x_test)
print(y_pred.size)
print(y_test.size)

print("plot results")
# invert the scaler to get the absolute price data
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)

# plots of prediction against actual data
plt.plot(y_test_orig, label='Actual Price', color='orange')
plt.plot(y_pred_orig, label='Predicted Price', color='green')

plt.title('BTC Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend(loc='best')

plt.show()
