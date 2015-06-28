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

import random 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
imgsize = 28
prt = 16
dgts_data = pd.read_csv("abcd.csv",index_col=0)
print dgts_data.head()
print dgts_data.shape
dgts_data = np.array(dgts_data)
print dgts_data.shape
#print dgts_data

dgts_lbl = pd.read_csv("abcd_l.csv",index_col=0)
#print dgts_lbl.head()
print dgts_lbl.shape
dgts_lbl = np.array(dgts_lbl)
print dgts_lbl.shape
#print dgts_lbl


pca = PCA(n_components=100)
pca.fit(dgts_data)
tr_dt_p = pca.transform(dgts_data)
print pca.explained_variance_ratio_
print tr_dt_p.shape
print sum(pca.explained_variance_ratio_)


mdl = knn()
gen_k_sets = StratifiedShuffleSplit(dgts_lbl, n_iter=1, test_size=0.4)


for train_index, test_index in gen_k_sets:   
    train_data, test_data = dgts_data[train_index], dgts_data[test_index]
    train_class, test_class = dgts_lbl[train_index], dgts_lbl[test_index]
td = pd.DataFrame(test_data)
td.to_csv('abcd_2.csv')
tc = pd.DataFrame(test_class)
tc.to_csv('abcd_2_l.csv')







