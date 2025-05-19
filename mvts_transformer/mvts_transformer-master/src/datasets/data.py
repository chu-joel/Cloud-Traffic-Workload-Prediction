from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
from itertools import repeat, chain

import numpy as np
import pandas as pd
from tqdm import tqdm
from sktime.utils import load_data

from datasets import utils

logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (
                NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class TSRegressionArchive(BaseData):
    """
    Dataset class for datasets included in:
        1) the Time Series Regression Archive (www.timeseriesregression.org), or
        2) the Time Series Classification Archive (www.timeseriesclassification.com)
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        # self.set_num_processes(n_proc=n_proc)

        self.config = config

        self.all_df, self.labels_df = self.load_all(
            root_dir, file_list=file_list, pattern=pattern)
        # all sample IDs (integer indices 0 ... num_samples-1)
        self.all_IDs = self.all_df.index.unique()

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        print(self.feature_df, 'self.feature_df')
        print(self.feature_names, 'self.feature_names')
        print(self.all_df, 'self.all_df')
        print(self.labels_df, 'self.labels_df')
        print(self.all_IDs, 'self.all_IDs')
        print(self.all_df.shape)
        print(self.all_IDs.shape)
        # exit()

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(
                root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(
                os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(
                filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(
            p) and p.endswith('.ts')]
        # if len(input_paths) == 0:
        # raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(
            input_paths[0])  # a single file contains dataset
        return all_df, labels_df

    def load_single(self, filepath):

        # Every row of the returned df corresponds to a sample;
        # every column is a pd.Series indexed by timestamp and corresponds to a different dimension (feature)
        if self.config['task'] == 'regression':
            df, labels = utils.load_from_tsfile_to_dataframe(
                filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
            labels_df = pd.DataFrame(labels, dtype=np.float32)
        elif self.config['task'] == 'classification':
            df, labels = load_data.load_from_tsfile_to_dataframe(
                filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
            labels = pd.Series(labels, dtype="category")
            self.class_names = labels.cat.categories
            # int8-32 gives an error when using nn.CrossEntropyLoss
            labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)
        else:  # e.g. imputation
            try:
                data = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                               replace_missing_vals_with='NaN')
                if isinstance(data, tuple):
                    df, labels = data
                else:
                    df = data
            except:
                df, _ = utils.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                            replace_missing_vals_with='NaN')
            labels_df = None

        # (num_samples, num_dimensions) array containing the length of each series
        lengths = df.applymap(lambda x: len(x)).values
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))
        # pd.set_option('display.max_columns', None)

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            logger.warning(
                "Not all time series dimensions have same length - will attempt to fix by subsampling first dimension...")
            # TODO: this addresses a very specific case (PPGDalia)
            df = df.applymap(subsample)

        if self.config['subsample_factor']:
            df = df.applymap(lambda x: subsample(
                x, limit=0, factor=self.config['subsample_factor']))

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            logger.warning("Not all samples have same length: maximum length set to {}".format(
                self.max_seq_len))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0]*[row])) for row in range(df.shape[0])), axis=0)
        


        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        print(df.head(30))
        print(labels_df.head())
        print(labels_df.shape)
        exit()
        return df, labels_df

# Implementation of the new Google Trace Cluster Data
# Need to find a way to load it.


class GTCData(BaseData):
    """
    Dataset class for datasets included in:
        1) the Time Series Regression Archive (www.timeseriesregression.org), or
        2) the Time Series Classification Archive (www.timeseriesclassification.com)
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):
        print("==============================")
        print(config)
        print("==============================")
        # self.set_num_processes(n_proc=n_proc)
        self.config = config
        self.max_seq_len = self.config["max_seq_len"]
        dimensions = config['dimensions']
        self.all_df, self.labels_df = self.load_all(
            root_dir, pattern=pattern, dimensions=dimensions)
        # all sample IDs (integer indices 0 ... num_samples-1)
        self.all_IDs = self.all_df.index.unique()

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df
        
        

        # print(self.feature_df, 'self.feature_df')
        # print(self.feature_names, 'self.feature_names')
        # print(self.feature_df, 'self.feature_df')
        # print(self.all_df, 'self.all_df')
        # print(self.labels_df, 'self.labels_df')
        # print(self.all_IDs, 'self.all_IDs')

    def load_all(self, root_dir,dimensions, pattern=None):
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
        columnNames = columnNames[:dimensions]  # Name of the columns do not matter. So we can reduce the list if we did feature selection

        FSPattern = "-FS-" if dimensions != 14 else "_"
        path = ""
        if 'TRAIN' in pattern:
            path = root_dir+'/CloudWorkload_TRAIN'+FSPattern+'SEQUENCED.txt'
        elif "VALIDATION" in pattern:
            path = root_dir+'/CloudWorkload_VALIDATION'+FSPattern+'SEQUENCED.txt'
        else:
            path = root_dir+'/CloudWorkload_TEST'+FSPattern+'SEQUENCED.txt'

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
            # Turn labels into list
            labels = splitLine[1].replace("\n","")
            labels = labels.split(', ')
            labels = np.array(labels)
            labels = [float(x) for x in labels]
            labelList.append(labels)

            stringList = splitLine[0].split()
            floatList=[float(x) for x in stringList]
            float_array = np.array(floatList)

            # If sequence not a full sequence length
            # This is due to preprocessing when rows were removed
            if(len(floatList)==0):
                continue
            if(len(floatList) != dimensions*self.max_seq_len):    
                reshapedData = np.reshape(float_array,(int(len(floatList)/dimensions), dimensions))
            else:
                reshapedData = np.reshape(float_array,(self.max_seq_len, dimensions))
            featureData.append(reshapedData)

        # List to store dataframes for efficiency when concatonating
        dataframes = []

        # Arrange Data into dataframe
        df = pd.DataFrame()
        for i in tqdm(range(len(featureData)), desc="Converting feature data to dataframe"):
            dataFrame = pd.DataFrame(featureData[i], columns=columnNames)
            dataFrame.index = [i] * len(featureData[i])
            df = pd.concat([df, dataFrame])

            if len(featureData)==1 or i % int(len(featureData)/100) == 0 or i == len(featureData)-1:
                dataframes.append(df)
                df = pd.DataFrame()
            df = pd.concat([df, dataFrame])
        
        df = pd.DataFrame()

        # Concatonate all dataFrames
        for data in tqdm(dataframes, desc="Merging sub dataframes"):
            df = pd.concat([df, data])

        # labelList = [float(x) for x in labelList]
        labels_df = pd.DataFrame(labelList)
        return df, labels_df
    
    # DEPRECATED
    # Used for sequence length 1
    # def load_all(self, root_dir, pattern=None):
    #     columnNames = [
    #         'mean_CPU',
    #         'canonical_memory',
    #         'assigned_memory',
    #         'unmapped_page_cache_memory_usage',
    #         'total_page',
    #         'maximum_memory',
    #         'mean_disk',
    #         'mean_local',
    #         'maximum_CPU',
    #         'maximum_disk',
    #         'CPI',
    #         'MAI',
    #         'sampled_CPU_usage',
    #         'duration'
    #     ]
    #     path = ""
    #     if pattern == 'TRAIN':
    #         path = root_dir+'CloudWorkload_TRAIN.csv'
    #     elif pattern =="VAL":
    #         path = root_dir+'CloudWorkload_VAL.csv'
    #     else:
    #         path = root_dir+'CloudWorkload_TEST.csv'

    #     df = pd.read_csv(
    #         path, names=columnNames)
        
    #     labels_df = df[['mean_CPU']]
    #     df.drop(['mean_CPU'], axis=1)
    #     df = df.astype('float')
    #     labels_df = labels_df.astype('float')

    #     return df, labels_df



data_factory = {
                'tsra': TSRegressionArchive,
                'gtc': GTCData}
