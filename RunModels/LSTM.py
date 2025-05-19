from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Bidirectional, Dropout, Dense, LSTM
import pandas as pd
from tensorflow import keras
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import os

randomSeed = 477
random.seed(randomSeed)
np.random.RandomState(randomSeed)
os.environ['PYTHONHASHSEED'] = str(randomSeed)
print("loading data")

scaler = MinMaxScaler()


def load_all(filename, dimensions, sequenceLength, test=False):
    print("Loading Data")
    initialList = []
    print("Reading file: ", filename)
    # Read into list for each event
    with open(filename, "r") as file:
        initialList = file.readlines()

    # Take label out of data using delimiter
    labelList = []
    featureData = []
    print("Separating label from feature data")
    for list in initialList:
        splitLine = list.split(":")
        labels = splitLine[1].replace("\n", "")
        labels = labels.split(', ')
        labels = np.array(labels)
        labels = [float(x) for x in labels]
        labelList.append(labels)

        # labelList.append(splitLine[1].replace("\n", ""))

        # Split features into sequences 13 dimensons (columns) 10 rows
        stringList = splitLine[0].split()
        floatList = [float(x) for x in stringList]
        float_array = np.array(floatList)
        # If sequence not a full sequence length
        # This is due to preprocessing when rows were removed
        if (len(floatList) == 0):
            continue
        if len(floatList) == dimensions:
            reshapedData = float_array
        elif (len(floatList) != dimensions*sequenceLength):
            reshapedData = np.reshape(
                float_array, (int(len(floatList)/dimensions), dimensions))
        else:
            reshapedData = np.reshape(
                float_array, (sequenceLength, dimensions))
        featureData.append(reshapedData)

    # Add list of data to a dataframe
    df = np.empty(shape=(len(featureData), sequenceLength, dimensions))
    if (test):
        df = np.empty(shape=(len(featureData), dimensions))
    for i in range(len(featureData)):
        df[i] = featureData[i]

    labels_df = np.array(labelList)

    df = df.astype(float)

    return df, labels_df


def buildModel(sequenceLength, dropout_rate, dimensions, horizon):
    # build a 3-layer LSTM with 2 Bidirectional layers with dropout
    print("Building Model")
    model = keras.Sequential()
    # Hidden Layer
    model.add(
        LSTM(sequenceLength, return_sequences=True,
             input_shape=(sequenceLength, dimensions))
    )

    model.add(Dropout(rate=dropout_rate))
    # Bidirectional allows for training of sequence data forwards and backwards

    model.add(
        Bidirectional(LSTM(
            100, activation='relu'), input_shape=(sequenceLength, dimensions))
    )

    model.add(Dense(units=horizon))
    return model


def makePrediction(model, testdf, testlabel, learning_rate, batchSize, dropout_rate, epochs):
    results = model.evaluate(testdf, testlabel)
    print(results)
    y_pred = model.predict(testdf)

    y_test_orig = testlabel
    y_pred_orig = y_pred

    print("\nParameters: lr= "+str(learning_rate)+" bs= " +
          str(batchSize)+" dp= "+str(dropout_rate))
    print("MSE = "+str(round(mean_squared_error(y_test_orig, y_pred_orig), 8))+" MAE = "+str(round(mean_absolute_error(
        y_test_orig, y_pred_orig), 8))+" RMSE = "+str(round(mean_squared_error(y_test_orig, y_pred_orig, squared=True), 8)))

    print("plot results")
    print(y_pred_orig)
    y_test_orig = y_test_orig.flatten()
    y_pred_orig = y_pred_orig.flatten()

    file = open('../Results/newResults.txt', 'a')

    file.write('epochs: '+str(epochs))
    file.write('\n')
    file.write(str(y_pred_orig))
    file.write(("\nMSE = "+str(round(mean_squared_error(y_test_orig, y_pred_orig), 8))+" MAE = "+str(round(mean_absolute_error(
        y_test_orig, y_pred_orig), 8))+" RMSE = "+str(round(mean_squared_error(y_test_orig, y_pred_orig, squared=True), 8))))

    # plots of prediction against actual data
    plt.plot(y_test_orig, label='Actual CPU%', color='orange')
    plt.plot(y_pred_orig, label='Predicted CPU%', color='green')

    plt.title('Workload Traffic Prediction')
    plt.xlabel('Time')
    plt.ylabel('CPU%')
    plt.legend(loc='best')
    # Save plot
    title = "30 seconds epochs:"+str(epochs)+" MSE = "+str(round(mean_squared_error(y_test_orig, y_pred_orig), 8))+" MAE = "+str(round(mean_absolute_error(
        y_test_orig, y_pred_orig), 8))+" RMSE = "+str(round(mean_squared_error(y_test_orig, y_pred_orig, squared=True), 8))
    file.close()
    plt.savefig(title+".png")
    # plt.show()


def runLSTM(batchSize, dropout_rate, learning_rate,
            sequenceLength, dimensions, epochs, horizon):

    scaler = MinMaxScaler()
    traindf, trainlabel = load_all(
        "CloudWorkload_TRAIN_SEQUENCED.txt", dimensions, sequenceLength)
    testdf, testlabel = load_all(
        "CloudWorkload_TEST_SEQUENCED.txt", dimensions, sequenceLength)

    validationdf, validationlabel = load_all(
        "CloudWorkload_VALIDATION_SEQUENCED.txt", dimensions, sequenceLength)
    traindf = traindf.reshape(
        traindf.shape[0], sequenceLength * dimensions)

    traindf = scaler.fit_transform(traindf)
    traindf = traindf.reshape(traindf.shape[0], sequenceLength, dimensions)
    model = buildModel(sequenceLength, dropout_rate, dimensions, horizon)
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    # Do training
    model.compile(loss='mse', optimizer='adam')
    model.fit(
        x=traindf,
        y=trainlabel,
        epochs=epochs,
        batch_size=batchSize,
        shuffle=False,
        validation_data=(validationdf, validationlabel),
        callbacks=[callback]
    )

    testdf = testdf.reshape(len(testlabel), sequenceLength * dimensions)
    testdf = scaler.transform(testdf)

    testdf = testdf.reshape(len(testlabel), sequenceLength, dimensions)

    makePrediction(model, testdf, testlabel, learning_rate,
                   batchSize, dropout_rate, epochs)
