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

for nb in range(1,46,1):    
    for nb2 in range(1,nb*3,1):
        mdl = knn(n_neighbors=nb2)
        dst_mdl = nn(n_neighbors=nb)
        overall_mis = 0
        #mdl = SVC(C=1.0)
        #mdl = rfc(n_estimators=100)
        #mdl = knn(n_neighbours=1)
        
        for train_index, test_index in gen_k_sets:   
            train_data, test_data = dt[train_index], dt[test_index]
            train_class, test_class = lbls[train_index], lbls[test_index]
            tr_dts =[]
            tr_clses=[]
            print
            for k in range(3):
                tr_dt_idx = np.where(train_class==k)[0]
               
                tr_dt = train_data[tr_dt_idx,:]
         
          
                tr_dts.append(tr_dt)
                tr_clses.append(train_class[tr_dt_idx])
                
                
        
            j = 0
            for i,td in enumerate(test_data):
                td = np.array(td)
                tst_class_act=test_class[i]
               
                #print td.shape
                for k in range(3):
                 
                    tr_dt = tr_dts[k]
                    tr_cls = tr_clses[k]
                    #print tr_dt.shape
                    dst_mdl.fit(tr_dt)
                    nmbrs= dst_mdl.kneighbors(td,return_distance=True)
          
                    if k ==0:
                        dts = np.array(tr_dt[nmbrs[1],:])
                        
                        
                        dts = dts.reshape((dts.shape[1],dts.shape[2]))
                       
                        t_dt = dts
                    
                        t_cls= np.array(tr_cls[nmbrs[1]])
                    else:
                        dts = np.array(tr_dt[nmbrs[1],:])
                       
                        dts = dts.reshape((dts.shape[1],dts.shape[2]))
                    
                        t_dt = np.vstack((t_dt,dts))
                        t_cls = np.vstack((t_cls,tr_cls[nmbrs[1]]))
        
                t_cls = np.ravel(t_cls)
             
                mdl.fit(t_dt,t_cls)
                tst_class_pred = mdl.predict(td)
              
                
                #tst_class_pred=0
                if tst_class_pred != tst_class_act:
                    j= j+ 1
                    print
                    print nb2
                    print i,j    
                    print td
                    #print t_dt
                    #mdl2.fit(t_dt,t_cls)
                    #print mdl2.predict(td)
                    print tst_class_pred
                    print tst_class_act
                   
            print
            print 'j'
            print j
            overall_mis = overall_mis + j
        print
        print 'nb'
        print nb,overall_mis,nb2
        print        
        ab.append((nb,overall_mis,nb2))
print ab
            
            
        
 