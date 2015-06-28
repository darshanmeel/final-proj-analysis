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
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as rfc
imgsize = 28
prt = 2
dgts_data = pd.read_csv("abcd.csv",index_col=0)
print dgts_data.head()
print dgts_data.shape
dgts_data = np.array(dgts_data)
print dgts_data.shape
#print dgts_data


for prt in range(3,4,1):        
    #add 16 by 16 as well
    dt_data = np.reshape(dgts_data,(dgts_data.shape[0],imgsize,imgsize))
    for i in range(imgsize-prt):
        rw1 = i
        rw2 = i + prt
        for j in range(imgsize-prt):
            col1 = j
            col2 = j + prt
            ar = np.reshape(dt_data[:,rw1:rw2,col1:col2],(dt_data.shape[0],prt*prt))
            ad= np.reshape(np.percentile(ar,q=50,axis=1),(dt_data.shape[0],1))
            dgts_data = np.hstack((dgts_data,ad))
          
    print dgts_data.shape
    cn_data = dgts_data[:,784:]
    print cn_data.shape
    dgts_data = dgts_data[:,0:784]
    print dgts_data.shape
    
    dgts_lbl = pd.read_csv("abcd_l.csv",index_col=0)
    #print dgts_lbl.head()
    print dgts_lbl.shape
    dgts_lbl = np.array(dgts_lbl)
    print dgts_lbl.shape
    #print dgts_lbl
    
    n_neighbours = range(1,2,1)
    print n_neighbours
    
    
    gen_k_sets = StratifiedShuffleSplit(dgts_lbl, n_iter=1, test_size=0.15)
    
    print prt
    cn_data=dgts_data
    for train_index, test_index in gen_k_sets:   
        train_data, test_data = cn_data[train_index], cn_data[test_index]
        train_class, test_class = dgts_lbl[train_index], dgts_lbl[test_index]
        train_class = np.array(np.ravel(np.array(train_class)))
        test_class = np.array(np.ravel(np.array(test_class)))
        for nb in n_neighbours:
            print 'nb',nb,prt
            mdl = knn(n_neighbors= nb)
            mdl = sklearn.svm.SVC()
            mdl.fit(train_data,train_class)
            print 'score'
            print mdl.score(test_data,test_class)
            print 
        #print mdl.feature_importances_





