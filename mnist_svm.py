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
iris_data = datasets.load_digits()
print iris_data
dt = iris_data.data
lbls = iris_data.target

print dt.shape

#train a KNN and see how does it perform. Keep 50000 for training and 10000 for validation and 10000 for final test.



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
#lbls = dgts_lbls
#dt = dgts_data
    
num_fold = 10
gen_k_sets = StratifiedKFold(lbls,num_fold,shuffle=True)
ab = []


for nb in range(1,2,1):    
    dst_mdl = nn(n_neighbors=nb)    
    for c in range(1,10,10):
        overall_mis = 0
        err=[]
        mdl = SVC(C=c,kernel='rbf',degree=1,tol=0.00001)
        mdl = rfc(n_estimators=500,criterion='entropy',min_samples_leaf=2,min_samples_split=5,max_features=20)
        #mdl = knn(n_neighbours=1)
        
        for train_index, test_index in gen_k_sets:   
            train_data, test_data = dt[train_index], dt[test_index]
            train_class, test_class = lbls[train_index], lbls[test_index]
            tr_dts =[]
            tr_clses=[]
            print
            for k in range(10):
                tr_dt_idx = np.where(train_class==k)[0]
               
                tr_dt = train_data[tr_dt_idx,:]
         
          
                tr_dts.append(tr_dt)
                tr_clses.append(train_class[tr_dt_idx])
                
                
        
            j = 0
            for i,td in enumerate(test_data):
                td = np.array(td)
                tst_class_act=test_class[i]
               
                #print td.shape
                for k in range(10):
                 
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
                    
                    print i,j    
                    #print td
                   
                    print tst_class_pred
                    print tst_class_act
                   
            print
            print 'j'
            print j
            er = j*1.0/test_data.shape[0]
            overall_mis = overall_mis + j
            err.append(er)
        print
        print 'nb'
        print nb,overall_mis
        print
            
        ab.append((nb,overall_mis,c,err))
print ab
print sum(ab[0][3])/num_fold
            
            
        
 