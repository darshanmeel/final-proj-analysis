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
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC
dgts_data = pd.read_csv("abcd.csv",index_col=0)
print dgts_data.head()
print dgts_data.shape
dgts_data = np.array(dgts_data)
print dgts_data.shape
print dgts_data

dgts_lbl = pd.read_csv("abcd_l.csv",index_col=0)
print dgts_lbl.head()
print dgts_lbl.shape
dgts_lbl = np.array(dgts_lbl)
print dgts_lbl.shape
print dgts_lbl
print

#train a KNN and see how does it perform. Keep 50000 for training and 10000 for validation and 10000 for final test.

gen_k_sets = StratifiedShuffleSplit(dgts_lbl, n_iter=1, test_size=0.10)
dst_mdl = nn(n_neighbors=15)

for train_index, test_index in gen_k_sets:   
    train_data, test_data = dgts_data[train_index], dgts_data[test_index]
    train_class, test_class = dgts_lbl[train_index], dgts_lbl[test_index]
    tr_dts =[]
    print
    for k in range(10):
        tr_dt_idx = np.where(train_class==k)[0]
        print 
        print tr_dt_idx
        print
        tr_dt = train_data[tr_dt_idx,:]
        print tr_dt.shape
        tr_dts.append(tr_dt)
        
        
    
    #print mdl.score(train_data,train_class)
    print train_data.shape
    j = 0
    for i,td in enumerate(test_data):
        td = np.array(td)
        tst_class_act=test_class[i]
        avg_dist = []
        nbrs = []
        #print td.shape
        for k in range(10):
            tr_dt = tr_dts[k]
            #print tr_dt.shape
            dst_mdl.fit(tr_dt)
            nmbrs= dst_mdl.kneighbors(td,return_distance=True)
            avg_dist.append(nmbrs[0].mean())
            nbrs.append(nmbrs[1])
  
        #print avg_dist
        tst_class_pred = np.argmin(np.array(avg_dist))
        #print tst_class_pred
        #tst_class_pred=0
        if tst_class_pred != tst_class_act:
            j= j+ 1
            print
            print i,j    
            print avg_dist
            print tst_class_pred
            print tst_class_act
            for k in range(10):
                nmbrs= list(nbrs[k])
                print train_class[nmbrs]
        if i%500==0:
            print i
        
    print j
            
            
        
 