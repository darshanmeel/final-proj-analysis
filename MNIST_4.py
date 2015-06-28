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

dt_0_idx = np.where(dgts_lbl==0)[0]#[0:2000]
print 

tr_dt = dgts_data[dt_0_idx,:]
print tr_dt.shape

mdl =  KMeans(n_clusters=2)
mdl.fit(tr_dt)
print mdl.labels_
print mdl.cluster_centers_


pca = PCA(n_components=10)
pca.fit(tr_dt)
tr_dt_p = pca.transform(tr_dt)
print pca.explained_variance_ratio_
print tr_dt_p.shape


plt.figure(1, figsize=(12, 12))
plt.clf()

plt.scatter(tr_dt_p[:,0], tr_dt_p[:, 1], c=mdl.labels_.astype(np.float))

plt.show()






