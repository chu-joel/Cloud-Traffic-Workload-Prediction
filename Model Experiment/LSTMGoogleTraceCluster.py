from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from keras.layers import Dense
# from keras.layers import Bidirectional
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from sklearn.model_selection import train_test_split
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
randomSeed = 477
# for i in range(1000):
# randomSeed = random.randint(0, 1000)
print(randomSeed)
random.seed(randomSeed)

np.random.RandomState(randomSeed)
tf.random.set_seed(randomSeed)
os.environ['PYTHONHASHSEED'] = str(randomSeed)
print("loading data")


# Read from full_task_usage_99.csv and full_task_usage_199.csv
# df1 = pd.read_csv('full_task_usage_99.csv')
# df2 = pd.read_csv('full_task_usage_199.csv')


# df = pd.concat([df1, df2], sort=False)

df = pd.read_csv('5SecondFull.csv')


# df.sort_values(by=['start_time'], ascending=True)
print(df.columns)

# print(df.size)
# df = df.groupby(df.start_time // 300).mean()
# print(df.size)

# minTime = df['start_time'].min()
# df["start_time"] = df["start_time"].apply(
#     lambda x: timedelta(seconds=((x-minTime)/1000000)).minutes)

# df["start_time"] = df["start_time"].apply(
#     lambda x: divmod(((x-minTime)/1000000), 10)[0])

# df["start_time"] = df["start_time"].apply(
#     lambda x: divmod(((x-minTime)/1000000), 30)[0])

# print(df.head())

# df = df.groupby(df['start_time']).mean()
# df['start_time'] = df['start_time'].astype('datetime64[ns]')

# times = pd.to_datetime(df.start_time)
# df.groupby([df['start_time'].dt.hour,
#            df['start_time'].dt.minute]).value_col.sum()

# Formatting the grids for display
rcParams['figure.figsize'] = 14, 8
sns.set(style='whitegrid', palette='muted', font_scale=1.5)


print(df.head())
# df = df[::-1]


print(df.size)
# df.plot(y='sampled_CPU_usage', title='Subset Data Clean',
#         ylabel='CPU Usage', xlabel='Time')
# plt.show()


scaler = MinMaxScaler()
# fit the format of the scaler -> convert shape from (1000, ) -> (1000, 1)
cpus = df.sampled_CPU_usage.values.reshape(-1, 1)
print(cpus.size)

scaled_cpus = scaler.fit_transform(cpus)

seq_len = 60
batch_size = 128
# dropout_rate = 0.08

dropout_rate = 0.01
# dropout_rate = round(random.uniform(0.01, 0.5), 2)

# learning_rates = [0.005, 0.01, 0.001]
# batch_sizes = [16, 32, 64, 128, 256]
# dropout_rates = [0.01, 0.05, 0.1, 0.5]


def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

# Split data into training and test data


# def get_train_test_sets(data, seq_len, train_frac):
#     sequences = split_into_sequences(data, seq_len)
#     n_train = int(sequences.shape[0] * train_frac)
#     x_train = sequences[:n_train, :-1, :]
#     y_train = sequences[:n_train, -1, :]
#     x_test = sequences[n_train:, :-1, :]
#     y_test = sequences[n_train:, -1, :]
#     return x_train, y_train, x_test, y_test

# learning_rate = 0.015


learning_rate = 0.001
# learning_rate = round(random.uniform(0.1, 0.001), 3)

labels_df = df[['mean_CPU']]
df.drop(['mean_CPU'], axis=1)

# x_train, y_train, x_test, y_test = get_train_test_sets(
#     cpus, seq_len, train_frac=0.95)

x_train, y_train = train_test_split(df, test_size=0.05, shuffle=False)
x_test, y_test = train_test_split(labels_df, test_size=0.05, shuffle=False)


# print(y_test[0])

# plots of prediction against actual data

y_train_orig = scaler.inverse_transform(y_train)
# plt.plot(y_train_orig, label='Actual CPU%',
#          color='orange')
# plt.title('Training Data')
# plt.show()
print(y_test.size)
y_test_orig = scaler.inverse_transform(y_test)
plt.plot(y_test_orig, label='Actual CPU%', color='orange')
plt.title('Test Data')
plt.xlabel('Time')
plt.ylabel('CPU%')
plt.legend(loc='best')
plt.show()
exit()

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
    epsilon=1e-07,
    weight_decay=1e-6,
    amsgrad=False,
    name="Adam"
)

model.compile(loss='mean_squared_error', optimizer=adam)

history = model.fit(
    x_train,
    y_train,
    epochs=90,
    batch_size=batch_size,
    shuffle=False,
    validation_split=0.2,
)

y_pred = model.predict(x_test)

print("\nParameters: lr= "+str(learning_rate)+" bs= " +
      str(batch_size)+" dp= "+str(dropout_rate))
print("MSE = "+str(round(mean_squared_error(y_test, y_pred), 4))+" MAE = "+str(round(mean_absolute_error(
    y_test, y_pred), 4))+" RMSE = "+str(round(mean_squared_error(y_test, y_pred, squared=True), 4)))
# if mean_squared_error(y_test, y_pred) <= 0.019:

print("plot results")
# invert the scaler to get the absolute price data
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)

# plots of prediction against actual data
plt.plot(y_test_orig, label='Actual CPU%', color='orange')
plt.plot(y_pred_orig, label='Predicted CPU%', color='green')

plt.title('Workload Traffic Prediction')
plt.xlabel('Time')
plt.ylabel('CPU%')
plt.legend(loc='best')

plt.show()
