import pandas as pd
import seaborn as sns
from pylab import rcParams
import os

df1 = pd.read_csv('full_task_usage_99.csv')
df2 = pd.read_csv('full_task_usage_199.csv')

df = pd.concat([df1, df2])
df = df.sort_values('start_time')

# Parameters
timeInterval = 5
sequenceLength = 10
slidingWindowRate = 1
runTransformer = True
traingAndTest = True
preprocessData = True


def doPreprocessData():
    minTime = df['start_time'].min()

    # Sample data into time intervals
    df["start_time"] = df["start_time"].apply(
        lambda x: divmod(((x-minTime)/1000000), timeInterval)[0])

    df = df.groupby(df['start_time']).mean()
    print(df.head(237))
    print(df.shape)

    # (772673, 14)
    rcParams['figure.figsize'] = 14, 8
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)

    # print(df.head())
    df = df[::-1]

    df = df.sort_values('start_time')
    # print(df.size)
    # df.plot(y='sampled_CPU_usage', title='Subset Data Clean',
    #         ylabel='CPU Usage', xlabel='Time')
    # plt.show()

    # Sliding window
    print(len(df))

    labels_df = df[['mean_CPU']]
    df.drop(['mean_CPU'], axis=1)
    df = df.astype('float')
    labels_df = labels_df.astype('float')
    sequences = []
    labels = []

    # Split data into sliding window with sequences
    # Each meanCPU label will have the data of sequence length
    for i in range(0, len(df) - sequenceLength, slidingWindowRate):
        seq = df[i:i+sequenceLength]
        # Label will be the next mean CPU to be predicted
        label = labels_df.iloc[int(i+sequenceLength)]['mean_CPU']
        seq = seq.values.tolist()
        sequences.append(seq)
        labels.append(label)

    # Split into training, validation and testing splits (proportions)
    training = 0.8075
    validation = 0.1425
    test = 0.05

    def splitData(array, training, validation, test):
        totalLength = len(array)
        trainingSplit = array[int(training*totalLength):]
        validationSplit = array[int(
            training*totalLength):int((training+validation)*totalLength)]
        testSplit = array[:int(test*totalLength)]
        return trainingSplit, validationSplit, testSplit

    trainingData, validationData, testData = splitData(
        sequences, training, validation, test)

    trainingLabel, validationLabel, testLabel = splitData(
        labels, training, validation, test)

    # Use spaces as delimeter for data points
    # There will be 14(dimensions) x 10(Sequence length)

    def formatDataframeList(dataFrames):
        return "".join([" ".join(map(str, row)) for row in dataFrames])

    # Use : as delimiter to show the label for the data
    # Use \n new line for each row of data

    def writeToFile(labels, dataFrame, fileName):
        file = open(fileName, "w")
        for i in range(len(labels)):
            label = str(labels[i])
            data = str(formatDataframeList(dataFrame[i]))

            writeBuffer = "%s:%s\n" % (data, label)
            file.write(writeBuffer)

        file.close()

    # Output to string to output
    writeToFile(trainingLabel, trainingData,
                "CloudWorkload_TRAIN_SEQUENCED.txt")
    writeToFile(validationLabel, validationData,
                "CloudWorkload_VALIDATION_SEQUENCED.txt")
    writeToFile(testLabel, testData,
                "CloudWorkload_TEST_SEQUENCED.txt")


# Main method
if __name__ == "__main__":
    if preprocessData:
        doPreprocessData()
