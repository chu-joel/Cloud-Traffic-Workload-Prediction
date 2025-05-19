from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd
import random
import os


df2 = pd.read_csv('CloudWorkload_TEST.csv')
