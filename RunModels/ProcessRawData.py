import pandas as pd
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pd.set_option('display.max_columns', None)

"""
This file will process raw data downloaded from google cluster trace
Reads csv.gz files from 0-199 out of 500 files.
Performs Data interpolation on missing data and removed rows with unreasonable CPI, MAI and CPU
Output this into two files full_task_usage_99.csv and full_task_usage_199.csv as well as the zipped versions of the CSV
"""

# This is how we will aggregate different columns
agg_functions = {'mean_CPU': 'mean',
                 'canonical_memory': 'sum',
                 'assigned_memory': 'sum',
                 'unmapped_page_cache_memory_usage': 'sum',
                 'total_page': 'sum',
                 'maximum_memory': 'max',
                 'mean_disk': 'mean',
                 'mean_local': 'mean',
                 'maximum_CPU': 'max',
                 'maximum_disk': 'max',
                 'CPI': 'mean',
                 'MAI': 'mean',
                 'sampled_CPU_usage': 'mean',
                 'duration': 'mean'}

Total = pd.DataFrame()


def readFile(folder):
    name = folder+'/part-'+str(i).zfill(5)+'-of-00500.csv.gz'
    print(name)
    df = pd.DataFrame()
    df = pd.read_csv(name, names=['start_time',
                                  'end_time',
                                  'job_ID',
                                  'task_index',
                                  'machine_id',
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
                                  'sample_portion',
                                  'aggregation_type',
                                  'sampled_CPU_usage'], compression='gzip',
                     error_bad_lines=False)
    return df


def outputFiles(df):
    df = Total
    df = df.sort_values(by="start_time", ascending=True)
    minTime = df["start_time"].min()
    print(minTime)

    df = df.groupby(df['start_time']).aggregate(agg_functions)
    name = "full_task_usage_"+str(i)
    df.to_csv(name+'.csv')
    df.to_csv(name+'.csv.gz', compression='gzip')
    print(df.size)
    df = pd.read_csv(name+'.csv.gz', index_col=[0], compression='gzip')
    print(df.head())


def preprocessRaw(df):
    df = df.astype(float)
    df = df.drop(columns=['machine_id', 'job_ID',
                 'task_index', 'sample_portion', 'aggregation_type'])

    # Interpolate NaN values to fill missing data
    df = df.interpolate(limit_direction='both')
    print(df.shape)

    df['duration'] = df.apply(
        lambda row: row.end_time - row.start_time, axis=1)
    df = df.drop(columns=['end_time'])

    # According to https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md
    # Filter out measurement with unreasonable CPI, MAI and sampled CPU usage
    df.drop(df[df['sampled_CPU_usage']
               < 0.001].index, inplace=True)

    df.drop(df[df['MAI']
               < 0.001].index, inplace=True)
    df.drop(df[df['CPI']
               > 40].index, inplace=True)
    return df


def processRaw():
    folder_selected = filedialog.askdirectory()

    # Dont use 199-500 for this reason vvv
    # "Disk-time-fraction data is only included in about the first 14 days, because of a change in our monitoring system."
    # Therefore only use 0-199 of data which is 14 weeks(200 files)
    for i in range(0, 200):
        df = readFile(folder_selected)
        df = preprocessRaw(df)

        Total = pd.concat([df, Total], sort=False)

        # Output this to the respective files
        if i == 99 or i == 199:
            outputFiles(df)
            Total = pd.DataFrame()
