
from datetime import timedelta
import random
import pandas as pd
from tqdm import tqdm
import time
import numpy as np

from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector

np.random.seed(10)
random.seed(10)


"""
Make requirements.txt file for this
Requires numpy=1.23.5
"""

columnNames = [
    'mean_CPU',
    'canonical_memory',
    'assigned_memory',
    'unmapped_page_cache_memory_usage',
    'total_page',
    'maximum_memory',
    'mean_disk',
    'mean_local',
    'maximum_CPU',
    'maximum_disk',
    'CPI',
    'MAI',
    'sampled_CPU_usage',
    'duration'
]
path = 'SWCloudWorkload_TRAIN_SEQUENCED.txt'

max_seq_len = 10


def performFS():
    initialList = []
    print("Reading file")
    # Read into list for each event
    with open(path, "r") as file:
        initialList = file.readlines()

    # Take label out of data using delimiter
    # Use sequence length of 1 when
    labelList = []
    featureData = []
    for list in tqdm(initialList, desc="Separating label from feature data"):
        splitLine = list.split(":")
        labels = splitLine[1].replace("\n", "")
        labels = labels.split(', ')
        labels = np.array(labels)
        labels = [float(x) for x in labels]
        labelList.append(labels)

        dimensions = len(columnNames)

        # Split features into sequences 13 dimensons (columns) 10 rows
        stringList = splitLine[0].split()
        floatList = [float(x) for x in stringList]
        floatList = floatList[-dimensions:]

        featureData.append(floatList)

    df = pd.DataFrame(featureData, columns=columnNames,
                      dtype=float)
    labels_df = pd.DataFrame(labelList, dtype=float)

    efs = ExhaustiveFeatureSelector(LinearRegression(
    ), min_features=6, max_features=14, scoring='neg_mean_squared_error', cv=10, print_progress=True)
    t0 = time.time()
    print("Performing Feature selection")
    efs = efs.fit(df, labels_df)
    print("Feature Selection time:", str(
        timedelta(seconds=int(time.time()-t0))))
    print(efs.best_idx_)
    print(efs.best_feature_names_)
    print(efs.best_score_)
    return efs.best_feature_names_


# This is from efs.best_idx_
bestIndexes = [0, 3, 4, 5, 7, 8, 9, 10, 13]
bestFeatures = ['mean_CPU', 'unmapped_page_cache_memory_usage', 'total_page',
                'maximum_memory', 'mean_local', 'maximum_CPU', 'maximum_disk', 'CPI', 'duration']
usePreviousResults = True


def doWrapper():
    # Output files with the appropriate features
    if usePreviousResults:
        return bestFeatures
    else:
        return performFS()
