import pandas as pd
from sklearn.model_selection import train_test_split

sequenceLength = 10

df1 = pd.read_csv('full_task_usage_99.csv')
df2 = pd.read_csv('full_task_usage_199.csv')

df = pd.concat([df1,df2], sort=False)



# Sequencing
print(len(df))


# train, test = train_test_split(df, test_size=0.5)


# train.to_csv('CloudWorkload_TRAIN.csv')
# test.to_csv('CloudWorkload_TEST.csv')