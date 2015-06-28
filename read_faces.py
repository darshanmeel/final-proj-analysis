
"""
Created on Sun Jun 21 13:01:38 2015

@author: Inpiron
"""

import os,sys
#import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from numpy import *

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
from sklearn.decomposition import TruncatedSVD
fil = "E:\lfw-funneled\lfw_funneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg"
fil= "C:/Users/Inpiron/Desktop/20150621_152543.jpg"
img = plt.imread(fil)
print img.shape
print 'grn'
plt.imshow(img)
plt.show()
img_gray = rgb2gray(img)
print img_gray.shape
plt.imshow(img,cmap=cm.gray)
plt.show()
print 'grey'
plt.imshow(img_gray)
plt.show()
print img_gray

gulli_image = rgb2gray(plt.imread(fil))
fil = "C:/Users/Inpiron/Desktop/20150621_152529.jpg"
sahil_image = rgb2gray(plt.imread(fil))

dif = np.not_equal(gulli_image,sahil_image)
print dif
gul_dif =  gulli_image[dif]
sahil_diff =  sahil_image[dif]
print gul_dif.shape
print sahil_diff.shape


U,s,V = linalg.svd(gulli_image)
g_image = np.dot(U, np.dot(np.diag(s), V))

print g_image
print g_image.shape

plt.imshow(g_image)
plt.show()




