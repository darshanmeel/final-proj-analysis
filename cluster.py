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

dts = datasets.load_iris()
#print dts
dt = dts.data
lbls = dts.target

print dt

print lbls

'''
dgts_data = pd.read_csv("abcd_2.csv",index_col=0)

dt = np.array(dgts_data)
print dt.shape


dgts_lbl = pd.read_csv("abcd_2_l.csv",index_col=0)

lbls = np.array(dgts_lbl)
print lbls.shape
'''
train_lbl,test_lbl,train_dt,test_dt = train_test_split(lbls,dt,test_size = 0.20,random_state=1299004)

print train_dt.shape
print test_dt.shape

clst = KMeans(n_clusters = 10,n_init=20,tol=0.00001,max_iter=500)
clst.fit(train_dt)
clsts_lbl = clst.labels_

print clsts_lbl
print train_lbl
from sklearn.svm import SVC
plt.figure(figsize=(12,10))
plt.scatter(train_dt[:,0],train_dt[:,1],s = 40,c = clsts_lbl)
plt.show()

fr = pd.DataFrame(np.hstack((np.reshape(train_lbl,(train_dt.shape[0],1)),np.reshape(train_lbl,(train_dt.shape[0],1)),np.reshape(clsts_lbl,(train_dt.shape[0],1)))))

fr.columns = ['q','tr_lbl','cls_lbl']
print fr
grpd = fr.groupby(['cls_lbl','tr_lbl'],as_index=False)
a =  grpd.count()
print a
grpd = fr.groupby(['cls_lbl'],as_index=False)
b = grpd.count()
print b
c = pd.merge(a,b,left_on = 'cls_lbl',right_on = 'cls_lbl')
c = np.array(c)
print c[:,2]
print c[:,3]*1.0
d= np.divide(c[:,2],(c[:,3]*1.0)).reshape(c.shape[0],1)
print d
c= c[:,(0,1)]
print c.shape

c= pd.DataFrame(np.hstack((c,d)))
print c
d= np.array(c.pivot(index = 0,columns = 1, values=2).fillna(0.0))

mdl = rfc(n_estimators=100,criterion='entropy')
#mdl = SVC(C=1.0,kernel='rbf',degree=1,tol=0.00001,probability=True)
mdl.fit(train_dt,train_lbl)
print mdl.score(test_dt,test_lbl)
prbs = mdl.predict_proba(test_dt)
#print prbs
print d
ar_1 = []
ar_2 = []
ar_3 = []
ar_4 = []
for i,acc_lbl in enumerate(test_lbl):
    rf_lbl = np.argmax(prbs[i,:])
    a =clst.predict(test_dt[i])
    d_m = d[a,:]

    d_m_1 = np.add(prbs[i,:],d_m)
    rf_clst_lbl = np.argmax(d_m_1)

    ar_1.append((acc_lbl,rf_lbl,rf_clst_lbl))
    if acc_lbl <> rf_lbl:
        if acc_lbl == rf_clst_lbl:
            ar_2.append((acc_lbl,rf_lbl,rf_clst_lbl))
        else:
            ar_3.append((acc_lbl,rf_lbl,rf_clst_lbl))
    else:
        if acc_lbl <> rf_clst_lbl:
            ar_4.append((acc_lbl,rf_lbl,rf_clst_lbl))
            
    '''
    print 
    print i
    print acc_lbl
    
    print rf_lbl
    print prbs[i,:]
    
    print a
    
    
    print d_m_1
    '''
    
print 'ar_1',len(ar_1)
print

#print np.array(ar_1)
print 'ar_2',len(ar_2)
print
#print np.array(ar_2)
print 'ar_3',len(ar_3)
print
#print np.array(ar_3)
print 'ar_4',len(ar_4)
print
#print np.array(ar_4)
    

    






