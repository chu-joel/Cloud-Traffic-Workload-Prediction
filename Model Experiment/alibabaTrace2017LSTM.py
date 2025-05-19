from concurrent.futures import ProcessPoolExecutor
import math
from keras.layers import Dense
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow import keras
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd
import random
import tensorflow as tf
import os
# randomSeed = 15
randomSeed = 1
# for i in range(1000):
# randomSeed = random.randint(0,1000)
print(randomSeed)
random.seed(randomSeed)

np.random.RandomState(randomSeed)
tf.random.set_seed(randomSeed)
os.environ['PYTHONHASHSEED'] = str(randomSeed)
# define a cpu-intensive task
print("loading data")

# Each entry is 30 seconds apart.
# Read from server_usage.csv
df = pd.read_csv('server_usage.csv', names=['timestamp', 'machine_id', 'cpus',
                                            'memory', 'disk', 'oneMinute', 'fiveMinute', 'fifteenMinute'], nrows=10000)

# Formatting the grids for display
rcParams['figure.figsize'] = 14, 8
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

scaler = MinMaxScaler()
# fit the format of the scaler -> convert shape from (1000, ) -> (1000, 1)
cpus = df.cpus.values.reshape(-1, 1)
scaled_cpus = scaler.fit_transform(cpus)

seq_len = 60
batch_size = 32
dropout_rate = 0.01

# learning_rates = [0.005, 0.01, 0.001]
# batch_sizes = [16, 32, 64, 128, 256]
# dropout_rates = [0.01, 0.05, 0.1, 0.5]


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


learning_rate = 0.01
x_train, y_train, x_test, y_test = get_train_test_sets(
    scaled_cpus, seq_len, train_frac=0.7)

# fraction of the input to drop; helps prevent overfitting

window_size = seq_len - 1

# build a 3-layer LSTM RNN
model = keras.Sequential()

# Hidden Layer
model.add(
    LSTM(window_size, return_sequences=True,
         input_shape=(window_size, x_train.shape[-1]))
)

model.add(
    LSTM(window_size, return_sequences=True,
         input_shape=(window_size, x_train.shape[-1]))
)

model.add(Dropout(rate=dropout_rate))
# Bidirectional allows for training of sequence data forwards and backwards
model.add(
    Bidirectional(LSTM((window_size * 2), return_sequences=True)
                  ))

model.add(Dropout(rate=dropout_rate))
model.add(
    Bidirectional(LSTM(
        window_size, return_sequences=False))
)

model.add(Dense(units=1))
# linear activation function: activation is proportional to the input
model.add(Activation('relu'))

adam = keras.optimizers.Adam(
    learning_rate=learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=None,
    decay=1e-6,
    amsgrad=False,
    name="Adam"
)

model.compile(loss='mean_squared_error', optimizer=adam)

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=batch_size,
    shuffle=False,
    validation_split=0.2,
)

y_pred = model.predict(x_test)

if mean_squared_error(y_test, y_pred) <= 0.1:
    print("\nParameters: lr= "+str(learning_rate)+" bs= " +
          str(batch_size)+" dp= "+str(dropout_rate))
    print("MSE = "+str(round(mean_squared_error(y_test, y_pred), 3))+" MAE = "+str(round(mean_absolute_error(
        y_test, y_pred), 3))+" RMSE = "+str(round(mean_squared_error(y_test, y_pred, squared=True), 3)))
    print("plot results")
    # invert the scaler to get the absolute price data
    y_test_orig = scaler.inverse_transform(y_test)
    y_pred_orig = scaler.inverse_transform(y_pred)

    # plots of prediction against actual data
    plt.plot(y_test_orig, label='Actual CPU%', color='orange')
    plt.plot(y_pred_orig, label='Predicted CPU%', color='green')

    plt.title('Traffic Prediction')
    plt.xlabel('Time')
    plt.ylabel('CPU%')
    plt.legend(loc='best')

    plt.show()
