
"""
Created on Sat Jun 27 13:23:22 2015

@author: Inpiron
"""
import cPickle
def unpickle(file):    
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
    
fl = '‪E:/MNIST/cifar-10-batches-py/data_batch_1'

ab = unpickle(fl)
print ab