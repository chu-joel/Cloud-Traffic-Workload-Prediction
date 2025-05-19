import random
from tqdm import tqdm
import RunTransformer
import PreprocessFilteredData
import PreprocessHorizonData
import Wrapper

# Need to fix these dependency problems VVVV

"""
What is going on???
"""

# Tasks
runTransformer = False
runTestTransformer = False
preprocessData = False
useMiniSet = False
doFeatureSelection = True


readRawData = False
runLSTM = False

preprocessHorizonData = False
preprocessHorizonDataNaiveValidation = False
performWeightedMovingAverage = False

runSlidingWindowTransformer = False
testSlidingWindowTransformer = True


"""
Best parameters for each horizon
horizon= 30 dModel= 64 lr=1e-05slidingrate= 5 attention heads= 8
horizon= 20 dModel= 16 lr=1e-05slidingrate= 5 attention heads= 8

horizon= 10 dModel= 64 lr=8e-06slidingrate= 5 attention heads= 8
horizon= 10 dModel= 16 lr=5e-05slidingrate= 5 attention heads= 8

horizon= 10 dModel= 16 lr=1e-05slidingrate= 5 attention heads= 8 encodinglayers= 1
Last epochs= 30
Total runtime: 0.0 hours, 21.0 minutes, 2.5449836254119873 seconds
Best training loss= 0.0007523682884448687 at epoch= 29
Best validation loss= 0.0018189305334067772 at epoch= 28
Best testing loss= 1.2316321411978511e-05 MAE= 0.0025986039634999297 at epoch= 27
BestTest = [0.015771195, 0.015498868, 0.016746495, 0.017536946, 0.013777003, 0.016515348, 0.016521456, 0.015926559, 0.018003533, 0.018738886]
Early stop test= [0.015810862, 0.016663127, 0.015286094, 0.01665892, 0.016231995, 0.016710324, 0.018217063, 0.017819291, 0.017824767, 0.018583214]
Early stop MSE = 1.344e-05 MAE = 0.00297315

"""


# Transformer parameters
# Parameters
timeInterval = 30
sequenceLength = 15
slidingWindowRate = 5
learningRate = 0.00001
epochs = 50  # Early stopping should never let it get to here
dimensions = 14
selectedFeatures = []

dropout = 0.1
headNum = 8  # 16
encodingLayers = 1  # Original is 3

dModel = 16  # 16
posEncoding = 'fixed'
patience = 1
horizon = 10  # 20,30


# For hyperparameter optimization


# Split into training, validation and testing splits (proportions)
training = 0.8625
validation = 0.1125
test = 0.025


if readRawData:
    import ProcessRawData
    ProcessRawData.processRaw()
if doFeatureSelection:
    selectedFeatures = Wrapper.doWrapper()
    dimensions = len(selectedFeatures)


if preprocessData:
    PreprocessFilteredData.doPreprocessData(
        timeInterval, useMiniSet, sequenceLength, training, validation, test, slidingWindowRate, selectedFeatures)

if preprocessHorizonDataNaiveValidation:
    slidingWindowRate = 1
    PreprocessHorizonData.doPreprocessData(
        timeInterval, useMiniSet, sequenceLength, slidingWindowRate, selectedFeatures, performWeightedMovingAverage, horizon, validation=True)

if runTransformer:
    file = open('../Results/newResults.txt', 'a')

    file.write('\n\nhorizon= ' + str(horizon) + ' dModel= '+str(dModel)+' lr='+str(learningRate) +
               'slidingrate= ' + str(slidingWindowRate) + ' attention heads= ' + str(headNum))
    file.close()
    slidingWindowRate = 1
    PreprocessHorizonData.doPreprocessData(
        timeInterval, useMiniSet, sequenceLength, slidingWindowRate, selectedFeatures, performWeightedMovingAverage, horizon, validation=True)
    RunTransformer.trainTransformer(sequenceLength, epochs, learningRate,
                                    timeInterval, dropout, headNum, encodingLayers, dModel, posEncoding, patience, dimensions)
if runTestTransformer:
    RunTransformer.testTransformer(sequenceLength, epochs, learningRate,
                                   timeInterval, dropout, headNum, encodingLayers, dModel, posEncoding, patience, dimensions)

if runLSTM:
    import LSTM
    batchSize = 128
    dropout_rate = 0.01
    learning_rate = 0.001
    sequenceLength = 10

    file = open('../Results/newResults.txt', 'a')

    file.write('\n\n LSTM: horizon= ' + str(horizon))
    file.close()

    epochs = 90
    PreprocessHorizonData.doPreprocessData(
        timeInterval, useMiniSet, sequenceLength, slidingWindowRate, selectedFeatures, performWeightedMovingAverage, horizon, validation=True)
    LSTM.runLSTM(batchSize, dropout_rate, learning_rate,
                 sequenceLength, dimensions, epochs, horizon)
if preprocessHorizonData:
    PreprocessHorizonData.doPreprocessData(
        timeInterval, useMiniSet, sequenceLength, slidingWindowRate, selectedFeatures, performWeightedMovingAverage, horizon, validation=False)
# USED FOR HYPERPARAMETER OPTIMIZATION
# with tqdm(total=81) as progressBar:

# while True:
# for encodingLayer in encodingLayers:
#     for sequenceLength in sequenceLengths:
    # horizon = random.choice(horizons)
    # dModel = random.choice(dModels)
    # headNum = random.choice(headNums)
    # encodingLayer = random.choice(encodingLayers)
    # learningRate = random.uniform(0.0001, 0.00001)

    # learningRate = '%s' % float('%.3g' % learningRate)
if preprocessHorizonData:
    PreprocessHorizonData.doPreprocessData(
        timeInterval, useMiniSet, sequenceLength, slidingWindowRate, selectedFeatures, performWeightedMovingAverage, horizon, validation=False)

if runSlidingWindowTransformer:
    file = open('../Results/newResults.txt', 'a')

    file.write('\n\nhorizon= ' + str(horizon) + ' dModel= '+str(dModel)+' lr='+str(learningRate) +
               'slidingrate= ' + str(slidingWindowRate) + ' attention heads= ' + str(headNum) + ' encodinglayers= ' + str(encodingLayers))
    file.close()
    RunTransformer.trainSlidingWindowTransformer(sequenceLength, epochs, learningRate,
                                                 timeInterval, dropout, headNum, encodingLayers, dModel, posEncoding, patience, dimensions)


if testSlidingWindowTransformer:
    RunTransformer.testSlidingWindowTransformer(sequenceLength, epochs, learningRate,
                                                timeInterval, dropout, headNum, encodingLayers, dModel, posEncoding, patience, dimensions)
