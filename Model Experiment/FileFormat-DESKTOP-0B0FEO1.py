import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams

df1 = pd.read_csv('full_task_usage_99.csv')
df2 = pd.read_csv('full_task_usage_199.csv')

df = pd.concat([df1, df2])
df = df.sort_values('start_time')
minTime = df['start_time'].min()

# 30 seconds
# df["start_time"] = df["start_time"].apply(
#     lambda x: divmod(((x-minTime)/1000000), 30)[0])

# 10 seconds
# df["start_time"] = df["start_time"].apply(
#     lambda x: divmod(((x-minTime)/1000000), 10)[0])

# 5 seconds
df["start_time"] = df["start_time"].apply(
    lambda x: divmod(((x-minTime)/1000000), 5)[0])

df = df.groupby(df['start_time']).mean()


rcParams['figure.figsize'] = 14, 8
sns.set(style='whitegrid', palette='muted', font_scale=1.5)


# print(df.head())
df = df[::-1]


print(df.size)
df.plot(y='sampled_CPU_usage', title='Subset Data Clean',
        ylabel='CPU Usage', xlabel='Time')
plt.show()


train, test = train_test_split(df, test_size=0.05, shuffle=False)

train, val = train_test_split(train, test_size=0.15, shuffle=False)

test.plot(y='sampled_CPU_usage', title='Test Data Clean',
        ylabel='CPU Usage', xlabel='Time')
plt.show()
exit()

print(train)
print(val)
print(test)
print(train.columns)
val.to_csv('CloudWorkload_VAL.csv', index=None)
train.to_csv('CloudWorkload_TRAIN.csv', index=None)
test.to_csv('CloudWorkload_TEST.csv', index=None)
