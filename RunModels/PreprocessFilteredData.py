import pandas as pd
import seaborn as sns
from pylab import rcParams

# splitData helper function to split the appropriate array according to the proportions being passed in
# Used for both labels and feature data


def splitData(array, training, validation, test):
    totalLength = len(array)
    trainingSplit = array[:int(training*totalLength)]
    validationSplit = array[int(
        training*totalLength):int((validation+training)*totalLength)]
    testSplit = array[int((validation+training)*totalLength):]
    return trainingSplit, validationSplit, testSplit


# split list of list into a string using whitespace as delimiter
def formatDataframeList(dataFrames):
    # Use spaces as delimeter for data points
    # There will be 13(dimensions) x 10(Sequence length)
    return " ".join([" ".join(map(str, row)) for row in dataFrames])


# Writes labels and feature data to the given filename
def writeToFile(labels, dataFrame, fileName):
    # Use : as delimiter to show the label for the data
    # Use \n new line for each row of data
    file = open(fileName, "w")
    for i in range(len(labels)):
        label = str(labels[i])
        data = str(formatDataframeList(dataFrame[i]))
        writeBuffer = "%s:%s\n" % (data, label)
        file.write(writeBuffer)
    file.close()


def doPreprocessData(timeInterval, useMiniSet, sequenceLength, training, validation, test, slidingWindowRate, FS=[]):
    doFeatureSelection = len(FS) != 0
    df1 = pd.read_csv('full_task_usage_99.csv')
    df2 = pd.read_csv('full_task_usage_199.csv')

    df = pd.concat([df1, df2])
    df = df.sort_values('start_time')
    minTime = df['start_time'].min()

    # Sample data into time intervals
    df["start_time"] = df["start_time"].apply(
        lambda x: divmod(((x-minTime)/1000000), timeInterval)[0])

    df = df.groupby(df['start_time']).mean()

    rcParams['figure.figsize'] = 14, 8
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)

    # print(df.head())
    df = df[::-1]

    df = df.sort_values('start_time')

    if useMiniSet:
        subset = int(df.shape[0] * 0.5)
        df = df[:subset]

    labels_df = df[['mean_CPU']]
    # Why drop this when we have access to it? Exactly, we shouldnt drop it
    # df = df.drop(['mean_CPU'], axis=1)
    df = df.astype('float')
    labels_df = labels_df.astype('float')
    if doFeatureSelection:  # Keep only selected features
        df = df[FS]

    sequences = []
    labels = []

    nonSQTestData = []
    nonSQTestLabels = []
    # Sliding window
    # Split data into sliding window with sequences
    # Each meanCPU label will have the data of sequence length
    for i in range(0, len(df) - sequenceLength, slidingWindowRate):
        seq = df.iloc[i:i+sequenceLength]
        # Label will be the next mean CPU to be predicted
        label = labels_df.iloc[int(i+sequenceLength)]['mean_CPU']
        if len(seq) is not sequenceLength:
            print(len(seq))
            # exit()
        seq = seq.values.tolist()
        sequences.append(seq)
        labels.append(label)
        # Creates a non sequenced test set
        # if i == int(len(df)*(1.0-test)):
        #     for item in range(i+sequenceLength+1, len(df)):
        #         nonSQTestData.append([df.iloc[item]])
        #         nonSQTestLabels.append(labels_df.iloc[item]['mean_CPU'])

    trainingData, validationData, testData = splitData(
        sequences, training, validation, test)

    trainingLabel, validationLabel, testLabel = splitData(
        labels, training, validation, test)
    # Different file pattern for FS
    FSPattern = "-FS-" if doFeatureSelection else "_"

    # Write Full File
    writeToFile(labels, sequences, "CloudWorkload_FULL.txt")

    # Output to string to output
    writeToFile(testLabel, testData,
                "CloudWorkload_TEST"+FSPattern+"SEQUENCED.txt")
    writeToFile(trainingLabel, trainingData,
                "CloudWorkload_TRAIN"+FSPattern+"SEQUENCED.txt")

    writeToFile(validationLabel, validationData,
                "CloudWorkload_VALIDATION"+FSPattern+"SEQUENCED.txt")

    # Not needed for now
    # writeToFile(nonSQTestLabels, nonSQTestData,
    #             "CloudWorkload_TEST.txt")

    # Split data for sliding window 0.95 train 0.5 test
    trainingData, testData, validationData = splitData(
        sequences, 0.975, 0.025, validation)

    trainingLabel, testLabel, validationLabel = splitData(
        sequences, 0.975, 0.025, validation)

    writeToFile(trainingLabel, trainingData,
                "SWCloudWorkload_TRAIN"+FSPattern+"SEQUENCED.txt")

    writeToFile(testLabel, testData,
                "SWCloudWorkload_TEST"+FSPattern+"SEQUENCED.txt")
