import matplotlib
import numpy as np
import pandas as pd
import os
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2

import sklearn
from sqlalchemy import false

import cv2
import numpy as np
from skimage.util import random_noise


a = np.array([40,2,25,4,5,6])
b = np.array([20,2,3,4,5,6])

#seed = np.random.randint(0, 10000)
#np.random.seed(seed)
#np.random.shuffle(a)
#np.random.shuffle(b)

s = np.arange(a.shape[0])
np.random.shuffle(s)

a = a[s]
b =b[s]

print(a)
print(b)