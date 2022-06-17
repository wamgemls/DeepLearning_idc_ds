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

#%matplotlib inline

imagePatches = glob('../IDC_regular_ps50_idx5/**/*.png', recursive=True)

for filename in imagePatches[0:10]:
    print(filename)

patternZero = '*class0.png'
patternOne = '*class1.png'

classZero = fnmatch.filter(imagePatches, patternZero)
classOne = fnmatch.filter(imagePatches, patternOne)

print("IDC (-)\n\n", classZero[0:5], '\n')
print("IDC (+)\n\n", classOne[0:5], '\n')

def read_img(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (50,50))

    return img

labels = 'IDC (-)', 'IDC (+)'
sizes = [len(classZero), len(classOne)]
colors =['lightskyblue', 'red']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=30)

plt.axis('equal')
plt.show()

def augmentation1(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (50,50))
    augmented_img = np.fliplr(img)

    return augmented_img

def augmentation2(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (50,50))
    augmented_img = np.flipud(img)

    return augmented_img

def noisy(img_name):
    img = cv2.imread(img_name)
    noisy = np.random.poisson(img/255.0*300)/300*255
    noisy = cv2.resize(img, (50,50))
    noisy = noisy.astype(np.uint8)

    return noisy

#load 


print('Hello World')
print(cv2.__version__)