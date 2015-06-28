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
import sklearn
import random 
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC

    




dgts_data = pd.read_csv("abcd_2.csv",index_col=0)

dt = np.array(dgts_data)
print dt.shape


dgts_lbl = pd.read_csv("abcd_2_l.csv",index_col=0)

lbls = np.array(dgts_lbl)
print lbls.shape

num_fold = 1

ab = []

print dt.shape
print lbls.shape
overall_mis = 0
err=[]
c= 1.0


ab= []

repl_fact = 9
j = 0

td_df = pd.read_csv('td.csv',index_col=0)
tc_df = pd.read_csv('tc.csv',index_col=0)
tsd_df = pd.read_csv('tsd.csv',index_col=0)
test_data_df = pd.read_csv('test_data.csv',index_col=0)
test_class_df = pd.read_csv('test_class.csv',index_col=0)

imgsize = 28

td = np.array(td_df)
tc = np.array(tc_df).reshape(td.shape[0],1)
tsd = np.array(tsd_df)
test_data = np.array(test_data_df)
print test_data.shape
test_data = np.reshape(test_data,(test_data.shape[0],imgsize,imgsize))
test_class = np.array(test_class_df)


dtsize = test_data.shape[0]

for nb in range(9,10,1):
    mdl = SVC(C=c,kernel='rbf',degree=1,tol=0.0001)
    mdl = rfc(n_estimators=100,criterion='entropy',min_samples_leaf=5,min_samples_split=10,max_features=8)
    mdl = knn(n_neighbors=nb)
    mdl.fit(td,tc)
    for i in range(dtsize):
    
        td_index = []
        for k in range(repl_fact):
            td_index.append( dtsize*k + i)
            
        tsd_1 = np.array(tsd[td_index,:])
     
        
        tst_class_act=test_class[i]
        
        ab = mdl.kneighbors(tsd_1,return_distance=True,n_neighbors=8)
        pos = ab[1]
        outcome = np.ravel(tc[pos])
        '''        
        print ab
        print 1
        print pos
        print 
        print outcome
     
        print 
        '''
        
        tst_class_pred_df = pd.DataFrame(outcome)
        #print tst_class_pred
        try:
            tst_class_pred_l = list(tst_class_pred_df.mode().iloc[0])
        except:
            tst_class_pred_l = list(tst_class_pred_df.iloc[0])
    
        
      
        tst_class_pred = tst_class_pred_l[0]
        #print tst_class_pred
        #print tst_class_act
        #print ghy
    
        if int(tst_class_pred) != int(tst_class_act):
           
            j= j + 1
            print
            
            print i,j    
            #print td
           
            print tst_class_pred
            print tst_class_act
            t_sh = test_data[i,:,:]
            plt.imshow(t_sh,cmap=cm.gray)
            plt.show()
        
    print
    print 'j'
    print j
    
    er = j*1.0/test_data.shape[0]
    overall_mis = overall_mis + j
    err.append((nb,er,overall_mis))



print
print 'nb'
print err
print sum(err[1])/10.0

    
        
 