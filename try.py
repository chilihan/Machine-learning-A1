
import time
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import multigrad

import pickle
import matplotlib.pyplot as plt

from numpy.random import uniform

from scipy.linalg import norm

# Global variables
dtype = 'float32'
eps = np.finfo(np.double).eps  # -- a small number


def load_evergreen(dtype=dtype):
    with open('evergreen.pkl') as f:
        train_set, val_set, test_set = pickle.load(f)
    print train_set, val_set, test_set
    train_X, train_y = train_set
    print train_set
