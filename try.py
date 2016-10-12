
import time
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import multigrad

import pickle
import matplotlib.pyplot as plt

from numpy.random import uniform

from scipy.linalg import norm

a = np.array([3,4,5,6,7])
print ("a",a, a.shape)
ar = np.reshape(a,(1,len(a))) #(1,5)
aT = np.transpose(ar)
print ("ar",ar, ar.shape)

b=np.array([1,2,3])
print ("b",b, b.shape)
br = np.reshape(b,(len(b),1)) #(3,1)
bT = np.transpose(br)
print("br",br, br.shape)
cr = np.dot(br,ar) #(3,5)
cT = np.dot(aT,bT)
# c = a*bT
print ("cr",cr,cr.shape)
print ("cT",cT,cT.shape)
