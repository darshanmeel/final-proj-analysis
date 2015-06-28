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
iris_data = datasets.load_iris()
print iris_data
dt = iris_data.data
lbls = iris_data.target


#train a KNN and see how does it perform. Keep 50000 for training and 10000 for validation and 10000 for final test.


num_fold = 10
gen_k_sets = StratifiedKFold(lbls,num_fold,True)
dst_mdl = nn(n_neighbors=18)
overall_mis = 0

for train_index, test_index in gen_k_sets:   
    train_data, test_data = dt[train_index], dt[test_index]
    train_class, test_class = lbls[train_index], lbls[test_index]
    tr_dts =[]
    print
    for k in range(3):
        tr_dt_idx = np.where(train_class==k)[0]
       
        tr_dt = train_data[tr_dt_idx,:]
  
        tr_dts.append(tr_dt)
        
        

    j = 0
    for i,td in enumerate(test_data):
        td = np.array(td)
        tst_class_act=test_class[i]
        avg_dist = []
        nbrs = []
        #print td.shape
        for k in range(3):
            tr_dt = tr_dts[k]
            #print tr_dt.shape
            dst_mdl.fit(tr_dt)
            nmbrs= dst_mdl.kneighbors(td,return_distance=True)
            avg_dist.append(nmbrs[0].mean())
            nbrs.append(nmbrs[1])
  
        #print avg_dist
        tst_class_pred = np.argmin(np.array(avg_dist))
      
        
        #tst_class_pred=0
        if tst_class_pred != tst_class_act:
            j= j+ 1
            print
            print i,j    
            print avg_dist
            print tst_class_pred
            print tst_class_act
           
    print
    print 'j'
    print j
    overall_mis = overall_mis + j
print
print overall_mis
            
            
        
 