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
dgts_data = pd.read_csv("abcd.csv",index_col=0)
print dgts_data.head()
print dgts_data.shape
dgts_data = np.array(dgts_data)
print dgts_data.shape
print dgts_data
'''
mn = np.reshape(np.mean(dgts_data,axis=1),(dgts_data.shape[0],1))
med = np.reshape(np.median(dgts_data,axis=1),(dgts_data.shape[0],1))
mx = np.reshape(np.max(dgts_data,axis=1),(dgts_data.shape[0],1))
#mod = np.mode(dgts_data,axis=1)
sd = np.reshape(np.std(dgts_data,axis=1),(dgts_data.shape[0],1))
print mn.shape
print med.shape
print sd.shape
print mx.shape

dgts_data = np.hstack((dgts_data,mn))
print dgts_data.shape
dgts_data = np.hstack((dgts_data,mx))
dgts_data = np.hstack((dgts_data,med))
dgts_data = np.hstack((dgts_data,sd))
'''
q= range(0,101,10)
perc = np.reshape(np.percentile(dgts_data,q,axis=1),(dgts_data.shape[0],len(q)))
#dgts_data = np.hstack((dgts_data,perc))

dgts_lbl = pd.read_csv("abcd_l.csv",index_col=0)
#print dgts_lbl.head()
print dgts_lbl.shape
dgts_lbl = np.array(dgts_lbl)
print dgts_lbl.shape
#print dgts_lbl

#train a KNN and see how does it perform. Keep 50000 for training and 10000 for validation and 10000 for final test.

gen_k_sets = StratifiedShuffleSplit(dgts_lbl, n_iter=1, test_size=0.15)
mdl = knn(n_neighbors=9)
dst_mdl = nn(n_neighbors=9)

for train_index, test_index in gen_k_sets:   
    train_data, test_data = dgts_data[train_index], dgts_data[test_index]
    train_class, test_class = dgts_lbl[train_index], dgts_lbl[test_index]
    train_class = np.reshape(np.ravel(train_class),(train_data.shape[0],1))
    print train_class.shape
  
    #test_data= test_data[:1000,]
    #test_class = np.reshape(test_class[:1000],(1000,1))
    print test_class.shape
    print train_data.shape
    
    mdl.fit(train_data,train_class)
    #print mdl.score(train_data,train_class)
    
    
    test_data_pred = mdl.predict(test_data)
    #print test_data_pred
    #print test_class
    dst_metric = mdl.kneighbors(test_data,return_distance=False)
    print mdl.score(test_data,test_class)
'''
print dst_metric
dst_mdl.fit(train_data)
for i,td in enumerate(test_data):
    td = np.array(td)
    test_data_pred = mdl.predict(td)
    if test_data_pred != test_class[i]:
        print 
        print i
        print
        print test_data_pred
        print test_class[i]
        print dst_metric[i]
  
        nbrs = dst_metric[i]
        imdata = np.reshape(td[:,0:784],(28,28))
        plt.imshow(imdata, cmap = cm.Greys_r)
        plt.show()
        
        for j in nbrs:
            print j
            print train_class[j]
            #print train_data[i]
            imdata = np.reshape(np.array(train_data[j][:,0:783]),(28,28))
            plt.imshow(imdata, cmap = cm.Greys_r)
            plt.show()

        print 
        print 9
        print 
        nbrs= dst_mdl.kneighbors(td,return_distance=False)
        for j in nbrs:
            print train_class[j]
            #print train_data[i]
        break
'''      
    
