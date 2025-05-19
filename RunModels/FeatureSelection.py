import datetime
import random
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import time
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler


np.random.seed(10)
random.seed(10)


"""
Make requirements.txt file for this
Requires numpy=1.23.5
Latest version of tensorflow
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
path = 'CloudWorkload_TRAIN_SEQUENCED.txt'

max_seq_len = 10

initialList = []
print("Reading file")
# Read into list for each event
with open(path, "r") as file:
    initialList = file.readlines()

# Take label out of data using delimiter
labelList = []
featureData = []
for list in tqdm(initialList, desc="Separating label from feature data"):

    splitLine = list.split(":")
    labelList.append(splitLine[1].replace("\n", ""))

    dimensions = 14

    # Split features into sequences 13 dimensons (columns) 10 rows
    stringList = splitLine[0].split()
    floatList = [float(x) for x in stringList]
    floatList = floatList[-14:]

    featureData.append(floatList)

# List to store dataframes for efficiency when concatonating
dataframes = []

df = pd.DataFrame(featureData, columns=columnNames,
                  dtype=float)


labels_df = pd.DataFrame(labelList, dtype=float)

scaler = StandardScaler()
print("Performing feature selection...")
t0 = time.time()
df = scaler.fit_transform(df)


logistic = SelectFromModel(LogisticRegression(
    penalty='l1', solver='liblinear', C=10, max_iter=100000))
logistic.fit(df, labelList)
print("Feature Selection time:", str(
    datetime.timedelta(seconds=int(time.time()-t0))))
print(logistic)
print(logistic.get_support())

print(logistic.estimator_.coef_)
selected = np.array(columnNames)[logistic.get_support()]
print(selected)


def doRFE():
    for i in range(10):
        random_state = random.randrange(10000)
        min_features_to_select = 1
        randomForestRegressor = RandomForestRegressor()
        cv = KFold(10, shuffle=True, random_state=random_state)
        rfecv = RFECV(
            estimator=randomForestRegressor,
            step=1,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=10
        )
        print("Performing feature selection...")
        t0 = time.time()
        rfecv = rfecv.fit(df, labels_df.values.ravel())
        print(f"Optimal number of features: {rfecv.n_features_}")
        print(rfecv)
        print(rfecv.ranking_)
        print("Feature Selection time:", str(
            datetime.timedelta(seconds=int(time.time()-t0))))

        print(rfecv.grid_scores_)
        print(rfecv.support_)
        print("Feature ranking: ", rfecv.ranking_)
        print("random state=", random_state)
        plt.plot(rfecv.grid_scores_)
        plt.show()
