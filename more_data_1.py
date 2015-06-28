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
iris_data = datasets.load_digits()

dt = iris_data.data
lbls = iris_data.target

print dt.shape

    




dgts_data = pd.read_csv("abcd_2.csv",index_col=0)

dt = np.array(dgts_data)
print dt.shape


dgts_lbl = pd.read_csv("abcd_2_l.csv",index_col=0)

lbls = np.array(dgts_lbl)
print lbls.shape

num_fold = 1
#gen_k_sets = StratifiedKFold(lbls,num_fold,shuffle=True)
gen_k_sets = StratifiedShuffleSplit(lbls,n_iter=num_fold,test_size=0.10)
ab = []
#dt, a_test, lbls, b_test = sklearn.cross_validation.train_test_split(dt, lbls, test_size=0.66)
for train_index, test_index in gen_k_sets:   
    dt, dtd = dt[train_index], dt[test_index]
    lbls, tcd = lbls[train_index], lbls[test_index]

print dt.shape
print lbls.shape
overall_mis = 0
err=[]
c= 1.0
mdl = SVC(C=c,kernel='rbf',degree=1,tol=0.0001)
mdl = rfc(n_estimators=100,criterion='entropy',min_samples_leaf=5,min_samples_split=10,max_features=8)
mdl = knn(n_neighbors=9)
imgsize = 28
patchsize = 26
ab= []
gen_k_sets = StratifiedShuffleSplit(lbls,n_iter=num_fold,test_size=0.15)
for train_index, test_index in gen_k_sets:   

    train_data, test_data = dt[train_index], dt[test_index]
    train_class, test_class = lbls[train_index], lbls[test_index]
    dtsize= train_data.shape[0]
    train_data = train_data.reshape(dtsize,imgsize,imgsize)
    
    c1 = train_data[:,0:patchsize,0:patchsize] 
    '''
    a= c1[0,:,:]
    print a.shape
    print a
    plt.imshow(a,cmap=cm.gray)
    plt.show()
    '''
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    '''
    c1_flip_sh = c1_flip.reshape(c1_flip.shape[0],patchsize,patchsize)
    print 'flip'
    print c1_flip_sh.shape
    a = c1_flip_sh[0,:,:]
    print a.shape
    print a
    plt.imshow(a,cmap=cm.gray)
    plt.show
    print 'original'
    plt.imshow(train_data[0,:,:],cmap=cm.gray)
    plt.show()
    print g
    '''
    c1 =c1.reshape(dtsize,patchsize*patchsize)
    td = np.array(c1)
    #td = np.vstack((td,c1_flip))
    tc = train_class
    #tc = np.vstack((tc,train_class))
    
    
    
    
    c1 = train_data[:,imgsize-patchsize:imgsize,0:patchsize]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    td = np.vstack((td,c1))
    #td = np.vstack((td,c1_flip))
    #tc = np.vstack((tc,train_class))
    tc = np.vstack((tc,train_class))
    
    
    c1 = train_data[:,0:patchsize,imgsize-patchsize:imgsize]    
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    td = np.vstack((td,c1))
    #td = np.vstack((td,c1_flip))
    #tc = np.vstack((tc,train_class))
    tc = np.vstack((tc,train_class))
    
    
    c1 = train_data[:,imgsize-patchsize:imgsize,imgsize -patchsize:imgsize]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    td = np.vstack((td,c1))
    #td = np.vstack((td,c1_flip))
    #tc = np.vstack((tc,train_class))
    tc = np.vstack((tc,train_class))
    
    
    im_ra = (imgsize - patchsize)/2 
    im_ra2 = (imgsize + patchsize)/2
    print im_ra,im_ra2
    
    

    c1 = train_data[:,im_ra:im_ra2,im_ra:im_ra2]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    td = np.vstack((td,c1))
    #td = np.vstack((td,c1_flip))
    #tc = np.vstack((tc,train_class))
    tc = np.vstack((tc,train_class))
    
    #left 
    c1 = train_data[:,im_ra:im_ra2,0:patchsize]
    print c1.shape
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    td = np.vstack((td,c1))
    #td = np.vstack((td,c1_flip))
    #tc = np.vstack((tc,train_class))
    tc = np.vstack((tc,train_class))
    
    #right
    c1 = train_data[:,im_ra:im_ra2,imgsize-patchsize:imgsize]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    td = np.vstack((td,c1))
    #td = np.vstack((td,c1_flip))
    #tc = np.vstack((tc,train_class))
    tc = np.vstack((tc,train_class))
    
    
    #up
    c1 = train_data[:,0:patchsize,im_ra:im_ra2]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    td = np.vstack((td,c1))
    #td = np.vstack((td,c1_flip))
    #tc = np.vstack((tc,train_class))
    tc = np.vstack((tc,train_class))
    
    #down
    c1 = train_data[:,imgsize-patchsize:imgsize,im_ra:im_ra2]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    td = np.vstack((td,c1))
    #td = np.vstack((td,c1_flip))
    #tc = np.vstack((tc,train_class))
    tc = np.vstack((tc,train_class))
    

    mdl.fit(td,tc)
    j = 0
    dtsize = test_data.shape[0]
    test_data = test_data.reshape(dtsize,imgsize,imgsize)
    
    c1 = test_data[:,0:patchsize,0:patchsize]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    c1 =c1.reshape(dtsize,patchsize*patchsize)
    tsd = np.array(c1)
    tsd = np.vstack((tsd,c1_flip))


    c1 = test_data[:,imgsize-patchsize:imgsize,0:patchsize]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    c1 =c1.reshape(dtsize,patchsize*patchsize)
    tsd = np.vstack((tsd,c1))
    tsd = np.vstack((tsd,c1_flip))
    
    

    c1 = test_data[:,0:patchsize,imgsize-patchsize:imgsize]    
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    c1 =c1.reshape(dtsize,patchsize*patchsize)
    tsd = np.vstack((tsd,c1))
    #tsd = np.vstack((tsd,c1_flip))
    
    
 
    c1 = test_data[:,imgsize-patchsize:imgsize,imgsize-patchsize:imgsize]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    c1 =c1.reshape(dtsize,patchsize*patchsize)
    tsd = np.vstack((tsd,c1))
    #tsd = np.vstack((tsd,c1_flip))


    
    c1 = test_data[:,im_ra:im_ra2,im_ra:im_ra2]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    c1 =c1.reshape(dtsize,patchsize*patchsize)
    tsd = np.vstack((tsd,c1))
    #tsd = np.vstack((tsd,c1_flip))
    
    #left
    
    c1 = test_data[:,im_ra:im_ra2,0:patchsize]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    c1 =c1.reshape(dtsize,patchsize*patchsize)
    tsd = np.vstack((tsd,c1))
    #tsd = np.vstack((tsd,c1_flip))
    
    
    #right
    c1 = test_data[:,im_ra:im_ra2,imgsize-patchsize:imgsize]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    c1 =c1.reshape(dtsize,patchsize*patchsize)
    tsd = np.vstack((tsd,c1))
    #tsd = np.vstack((tsd,c1_flip))
    
    #up
    
    c1 = test_data[:,0:patchsize,im_ra:im_ra2]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    c1 =c1.reshape(dtsize,patchsize*patchsize)
    tsd = np.vstack((tsd,c1))
    #tsd = np.vstack((tsd,c1_flip))
    
    #down 
    c1 = test_data[:,imgsize-patchsize:imgsize,im_ra:im_ra2]
    for i in range(c1.shape[0]):
        c_o = c1[i,:,:]

        c_o_flip = np.fliplr(c_o).reshape(1,patchsize*patchsize)
        
        if i==0:
            c1_flip= c_o_flip
        else:
            c1_flip = np.vstack((c1_flip,c_o_flip))
    c1 = c1.reshape(dtsize,patchsize*patchsize)
    c1 =c1.reshape(dtsize,patchsize*patchsize)
    tsd = np.vstack((tsd,c1))
    #tsd = np.vstack((tsd,c1_flip))
    
 
 
    repl_fact = 9
    j = 0
    
    for i in range(test_data.shape[0]):

        td_index = []
        for k in range(repl_fact):
            td_index.append( dtsize*k + i)
            
        tsd_1 = np.array(tsd[td_index,:])
 
        
        tst_class_act=test_class[i]
        tst_class_pred_df = pd.DataFrame(mdl.predict(tsd_1))
        #print tst_class_pred
        try:
            tst_class_pred_l = list(tst_class_pred_df.mode().iloc[0])
        except:
            tst_class_pred_l = list(tst_class_pred_df.iloc[0])
        #print tst_class_pred

        
      
        tst_class_pred = tst_class_pred_l[0]
    
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
    err.append(er)
    

print
print 'nb'
print overall_mis,err
print sum(err)*100.0/num_fold
    
        
 