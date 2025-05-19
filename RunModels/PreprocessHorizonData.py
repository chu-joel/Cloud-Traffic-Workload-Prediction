import pandas as pd
import seaborn as sns
from pylab import rcParams

# splitData helper function to split the appropriate array according to the proportions being passed in
# Used for both labels and feature data


def splitData(array, horizon, split):
    testLength = 0
    # Using one data point for prediction. The rest is for training
    totalLength = len(array)
    trainingSplit = array[:int(int(totalLength-1-horizon-testLength)*split)]
    validationSplit = array[int(int(totalLength-1-horizon-testLength)*split):]
    return trainingSplit, validationSplit


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

        label = str(labels[i].tolist()).strip("[]")
        data = str(formatDataframeList(dataFrame[i]))
        writeBuffer = "%s:%s\n" % (data, label)
        file.write(writeBuffer)
    file.close()


def doPreprocessData(timeInterval, useMiniSet, sequenceLength, slidingWindowRate, FS=[], performWeightedMovingAverage=False, horizon=30, validation=False):
    doFeatureSelection = len(FS) != 0
    df1 = pd.read_csv('full_task_usage_99.csv')
    df2 = pd.read_csv('full_task_usage_199.csv')

    df = pd.concat([df1, df2])
    df = df.sort_values('start_time')
    minTime = df['start_time'].min()

    # if performWeightedMovingAverage:
    #     df = ApplyWeightedMovingAverage(df)

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
        subset = int(df.shape[0] * 0.2)
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
    validationSequences = []
    validationLabels = []
    testSequences = []
    testLabels = []
    testLength = 1
    # Sliding window
    # Split data into sliding window with sequences
    # Each meanCPU label will have the data of sequence length
    for i in range(0, len(df) - sequenceLength - horizon - testLength, slidingWindowRate):
        validationIndex = i+1
        valSeq = df.iloc[validationIndex:validationIndex+sequenceLength]
        valSeq = valSeq.values.tolist()
        valLabel = labels_df.iloc[int(
            validationIndex+sequenceLength): int(validationIndex+sequenceLength+horizon)]['mean_CPU']
        validationSequences.append(valSeq)
        validationLabels.append(valLabel)

        seq = df.iloc[i:i+sequenceLength]
        # Label will be the next mean CPU to be predicted
        label = labels_df.iloc[int(i+sequenceLength): int(i+sequenceLength+horizon)]['mean_CPU']

        seq = seq.values.tolist()
        sequences.append(seq)
        labels.append(label)

    for i in range(len(df) - sequenceLength - horizon - testLength, len(df) - sequenceLength - horizon):
        seq = df.iloc[i:i+sequenceLength]
        # Label will be the next mean CPU to be predicted
        label = labels_df.iloc[int(i+sequenceLength): int(i+sequenceLength+horizon)]['mean_CPU']

        seq = seq.values.tolist()
        testSequences.append(seq)
        testLabels.append(label)

    # Different file pattern for FS
    FSPattern = "-FS-" if doFeatureSelection else "_"

    # Split data for sliding window 0.90 train 0.1 validation

    if validation:
        sequences, validationSequences = splitData(sequences, horizon, 0.9)
        labels, validationLabels = splitData(labels, horizon, 0.9)
        writeToFile(testLabels, testSequences,
                    "CloudWorkload_TEST"+FSPattern+"SEQUENCED.txt")
        writeToFile(labels, sequences,
                    "CloudWorkload_TRAIN"+FSPattern+"SEQUENCED.txt")

        writeToFile(validationLabels, validationSequences,
                    "CloudWorkload_VALIDATION"+FSPattern+"SEQUENCED.txt")
        return

    # Write Full File
    writeToFile(labels, sequences, "CloudWorkload_FULL.txt")

    writeToFile(validationLabels, validationSequences,
                "SWCloudWorkload_VAL"+FSPattern+"SEQUENCED.txt")

    writeToFile(labels, sequences,
                "SWCloudWorkload_TRAIN"+FSPattern+"SEQUENCED.txt")

    writeToFile(testLabels, testSequences,
                "SWCloudWorkload_TEST"+FSPattern+"SEQUENCED.txt")
