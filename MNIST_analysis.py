# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:50:31 2015
@author: dsing001
"""

from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn import datasets 
import math
import datetime
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import NearestNeighbors as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pandas import DataFrame as df
import random 
dgts_data = pd.read_csv("abcd.csv",index_col=0)
print dgts_data.head()
print dgts_data.shape


dgts_lbl = pd.read_csv("abcd_l.csv",index_col=0)

print dgts_lbl.head()
print dgts_lbl.shape


dgts = df.merge(dgts_data,dgts_lbl,right_index=True,left_index=True)

print dgts.head()
print dgts.shape
