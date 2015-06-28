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

dt = np.array(dgts_data)


    
dgts_lbl = pd.read_csv("abcd_l.csv",index_col=0)

lbls = np.array(dgts_lbl)

#print dgts_lbl

iris_data = datasets.load_digits()

#dt = iris_data.data
sc= sklearn.preprocessing.StandardScaler(copy=False)

#lbls = iris_data.target

print dt.shape
print lbls.shape

n_neighbours = range(1,2,1)
print n_neighbours

add_knn= True
n_folds=1
gen_k_sets = StratifiedShuffleSplit(lbls, n_iter=n_folds, test_size=0.15)
#gen_k_sets = StratifiedKFold(lbls,n_folds=10,shuffle=True)
ab= []
for nb in n_neighbours:
    bc =[]
    for train_index, test_index in gen_k_sets:   
        train_data, test_data = dt[train_index], dt[test_index]
        td= train_data
        tsd = test_data
        train_class, test_class = lbls[train_index], lbls[test_index]
        train_class = np.array(np.ravel(np.array(train_class)))
        test_class = np.array(np.ravel(np.array(test_class)))
       
        mdl = knn(n_neighbors=1)
        mdl.fit(train_data,train_class)    
    
        f_c = np.array(mdl.predict(test_data)).reshape((test_data.shape[0],1))
    
        test_data = np.hstack((test_data,f_c))
        
        
        mdl = nn(n_neighbors = 2)
        mdl.fit(train_data)
        
        nbrs= mdl.kneighbors(train_data,return_distance=True)[1][:,1]
        
        
        f_c = np.array(train_class[nbrs]).reshape((train_data.shape[0],1))
      
        train_data = np.hstack((train_data,f_c))
    
    
        if add_knn==False:
            train_data = train_data[:,0:-1]
            test_data = test_data[:,0:-1]
        sc.fit_transform(train_data)
     
        sc.transform(test_data)
        print test_data.shape,train_data.shape
        
        print 'nb',nb
        mdl = knn(n_neighbors= nb)
        mdl = sklearn.svm.SVC(kernel='linear',C=10.0)
        mdl=rfc(n_estimators=500,criterion='entropy')
        mdl.fit(train_data,train_class)
       
        bc.append(mdl.score(test_data,test_class))
        print 
    ab.append((nb,bc))
print ab
print
print sum(ab[0][1])/n_folds






