# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:16:31 2015

@author: Inpiron
"""

import sklearn
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cross_validation import train_test_split 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as knn
dts = datasets.load_iris()
#print dts
dt = dts.data
lbls = dts.target


dgts_data = pd.read_csv("abcd.csv",index_col=0)

dt = np.array(dgts_data)
print dt.shape


dgts_lbl = pd.read_csv("abcd_l.csv",index_col=0)

lbls = np.array(dgts_lbl)
print lbls.shape

train_lbl,test_lbl,train_dt,test_dt = train_test_split(lbls,dt,test_size = 0.15,random_state=1299004)

clstrs = 10
clst = KMeans(n_clusters = clstrs,n_init=30,tol=0.00001,max_iter=500)
clst.fit(train_dt)
clsts_lbl = np.reshape(np.array(clst.labels_),(train_dt.shape[0],1))

#td = np.hstack((train_dt,clsts_lbl))
mdls = []

for i in range(clstrs):
    t_idx = np.where(clsts_lbl==i)[0]
    mdl = rfc(n_estimators=50,criterion='entropy',oob_score=True)
    mdl = etc(n_estimators=5000,criterion='entropy',oob_score=True,bootstrap=True,min_samples_split=30)
    #mdl = SVC(C=10000,gamma=0.00001,kernel='rbf')
    mdl = knn(n_neighbors=1)
    td = train_dt[t_idx]
    tc = train_lbl[t_idx]
    #print tc
    mdl.fit(td,tc)
    print mdl.score(td,tc)

    #print mdl.oob_score_
    mdls.append(mdl)
    
scrs= []
clst_lbl_tst = np.reshape(np.array(clst.predict(test_dt)),(test_dt.shape[0],1))
for i in range(clstrs):
    t_idx = np.where(clst_lbl_tst==i)[0]
    tsd = test_dt[t_idx]
    tsc = test_lbl[t_idx]
    #print tsc
    mdl = mdls[i]
    #print mdl
    scr = mdl.score(tsd,tsc)
    scrs.append(scr)
    prdct = mdl.predict(tsd)
    #print prdct
    print i,scr,tsd.shape

print scrs