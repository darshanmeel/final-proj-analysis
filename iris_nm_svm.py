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
gen_k_sets = StratifiedKFold(lbls,num_fold)
ab = []
for nb in range(1,136,1):
    
    dst_mdl = nn(n_neighbors=nb)
    overall_mis = 0
    mdl = SVC(C=1.0,kernel='linear')
    mdl=rfc(n_estimators=500)
  
    

    for train_index, test_index in gen_k_sets:   
        train_data, test_data = dt[train_index], dt[test_index]
        train_class, test_class = lbls[train_index], lbls[test_index]
    
    
        j = 0
        


        for i,td in enumerate(test_data):
            td = np.array(td)
            tst_class_act=test_class[i]
            dst_mdl.fit(train_data)
            
            nmbrs= dst_mdl.kneighbors(td,return_distance=True)[1]
            print nmbrs.shape
            nmbrs = nmbrs.reshape((nb))
            print 1
            print
            train_data = train_data[nmbrs,:]
            train_class = train_class[nmbrs]
            if len(np.unique(train_class))==1:
                tst_class_pred = train_class[0]
            else:
                print train_data.shape
                print train_class.shape
                mdl.fit(train_data,train_class)
                tst_class_pred = mdl.predict(td)
          
            
            #tst_class_pred=0
            if tst_class_pred != tst_class_act:
                j= j+ 1
                print 'mis'
                print i,j    
          
                print tst_class_pred
                print tst_class_act
               
        print
        print 'j'
        print j
        overall_mis = overall_mis + j

    print
    print nb,overall_mis
    ab.append((nb,overall_mis))
print ab 
            
        
 