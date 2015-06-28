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
import random 

mnist = fetch_mldata('MNIST original')

dgts_data = mnist.data
dgts_labels = mnist.target

dt = pd.DataFrame(dgts_data)
dt.to_csv('abcd.csv')
lbls = pd.DataFrame(dgts_labels)
lbls.to_csv('abcd_l.csv')


