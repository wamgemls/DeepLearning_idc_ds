from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from glob import glob
import fnmatch
import matplotlib.pylab as plt
import cv2
import scipy
import pandas
import imblearn
import random
import tensorflow_datasets as tfds
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
#from sqlalchemy import false



"""
Balancing Strategy:
$0 = Manual Oversampling
$1 = imblearn Package

"""
Balancing_strategy = 1
sample_size = 500

if (sample_size % 2):
        sample_size = sample_size-1

halfsample=int(sample_size/2) # TwoClasses -> 50/50 dataset

# Data aquisition/Analysis

image_raw = glob('../IDC_regular_ps50_idx5/**/*.png', recursive=True)

for filename in image_raw[0:10]:
    print(filename)

patternZero = '*class0.png'
patternOne = '*class1.png'

file_classZero = fnmatch.filter(image_raw, patternZero)
file_classOne = fnmatch.filter(image_raw, patternOne)

print("IDC (-)\n\n", file_classZero[0:5], '\n')
print("IDC (+)\n\n", file_classOne[0:5], '\n')

def read_img(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (50,50))
    return img

labels = 'IDC (-) ' + str(len(file_classZero)) , 'IDC (+) ' + str(len(file_classOne))
sizes = [len(file_classZero), len(file_classOne)]
colors =['lightskyblue', 'red']


plt.figure()
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=30)
plt.axis('equal')


#Image Processing


# Data Preprocessing
# Balancing Method A (Manual manualoversamp Class 1)

def flip1(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (50,50))
    augmented_img = np.fliplr(img)
    return augmented_img

def flip2(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (50,50))
    augmented_img = np.flipud(img)
    return augmented_img

def noisy(img_name):
    img = cv2.imread(img_name)
    noisy = np.random.poisson(img / 255.0 * 400) / 400 * 255
    noisy = cv2.resize(noisy, (50,50))
    noisy = noisy.astype(np.uint8)
    return noisy

def file2img(filelist):

    DS_img = []

    for file in filelist:
        DS_img.append(read_img(file))
        
    return DS_img

#augmentation/noisy test

image_name = '../IDC_regular_ps50_idx5/8913/1/8913_idx5_x301_y1151_class1.png'

image = cv2.imread(image_name)
image = cv2.resize(image, (50,50))
#plt.figure()
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

image = flip1(image_name)
#plt.figure()
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

image = flip2(image_name)
#plt.figure()
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

image = noisy(image_name)
#plt.figure()
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

augmentlist = [flip1, flip2, noisy]



def datasetgen(classZero, classOne):

    DS_img = []
    DS_lbl = []
    DS_img.extend(classZero)
    DS_img.extend(classOne)

    DS_lbl.extend(np.zeros(len(classZero)))
    DS_lbl.extend(np.ones(len(classOne)))

    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    np.random.shuffle(DS_img)
    np.random.shuffle(DS_lbl)

    return DS_img, DS_lbl

def manualoversamp(file_osclass,file_nclass):
    
    disbalance = len(file_nclass) - len(file_osclass)
    os_images=[]
    
    for i in range(disbalance):

        augfunc = random.choice(augmentlist)
        random_image= random.choice(file_osclass)
        img = augfunc(random_image)
        os_images.append(img)

    return os_images


if(Balancing_strategy==0):
    
    file_classZero = file_classZero[0:800]
    file_classOne = file_classOne[0:300]

    if halfsample <= len(file_classZero) and halfsample <= len(file_classOne):
        file_classZero = file_classZero[0:halfsample]
        file_classOne = file_classOne[0:halfsample]

        img_classZero = file2img(file_classZero)
        img_classOne = file2img(file_classOne)
        
        DS = datasetgen(img_classZero, img_classOne)


    elif halfsample <= len(file_classZero) and halfsample > len(file_classOne):
        #manualoversamp ClassOne
        img_classZero = file2img(file_classZero[0:halfsample])
        img_classOne = file2img(file_classOne)
        
        os_data = manualoversamp(file_classOne,file_classZero[0:halfsample])
        img_classOne.extend(os_data)
        
        DS = datasetgen(img_classZero, img_classOne)


    elif halfsample > len(file_classZero) and halfsample <= len(file_classOne):
        #manualoversamp ClassZero
        img_classZero = file2img(file_classZero[0:halfsample])
        img_classOne = file2img(file_classOne)
        
        os_data = manualoversamp(file_classZero,file_classOne[0:halfsample])
        img_classZero.extend(os_data)
        
        DS = datasetgen(img_classZero, img_classOne)

    else:
        print('not enough data')

elif (Balancing_strategy==1):

    img_classZero = file2img(file_classZero[0:halfsample])
    img_classOne = file2img(file_classOne[0:halfsample])

    DS = datasetgen(img_classZero, img_classOne)






DS_img=np.array(DS[0])
DS_img=DS_img/255.0
DS_lbl=np.array(DS[1])

X_train, X_test, Y_train, Y_test = train_test_split(DS_img,DS_lbl,test_size=0.2)

Y_trainHot = to_categorical(Y_train, num_classes = 2)
Y_testHot = to_categorical(Y_test, num_classes = 2)

print(X_train.shape)
print(X_test.shape)


print(Y_train.shape)
print(Y_test.shape)
print(Y_trainHot.shape)
print(Y_testHot.shape)


#Final Inspection

imgs0 = DS_img
imgs1 = DS_img
imgs0[DS_lbl==0] = 1
imgs0[DS_lbl==1] = 0
imgs1[DS_lbl==1] = 1
imgs1[DS_lbl==0] = 0

print(np.sum(imgs0))
print(np.sum(imgs1))

# Deal with imbalanced class sizes below
# Make Data 1D for compatability upsampling methods
X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)
print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("X_trainFlat Shape: ",X_trainFlat.shape)
print("X_testFlat Shape: ",X_testFlat.shape)


ros = RandomOverSampler(sampling_strategy='auto')
#ros = RandomUnderSampler(sampling_strategy='auto')
X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_trainRosHot = to_categorical(Y_trainRos, num_classes = 2)
Y_testRosHot = to_categorical(Y_testRos, num_classes = 2)
#print("X_train: ", X_train.shape)
#print("X_trainFlat: ", X_train.shape)
print("X_trainRos Shape: ",X_trainRos.shape)
print("X_testRos Shape: ",X_testRos.shape)
print("Y_trainRosHot Shape: ",Y_trainRosHot.shape)
print("Y_testRosHot Shape: ",Y_testRosHot.shape)


"""
for row in range(3):

    plt.figure(figsize=(20,10))
    for col in range(3):
        plt.subplot(1,8,col+1)
"""
    






def func_test_1():
    print ("This is func_test_1.")

def func_test_2():
    print ("This is func_test_2.")

def func_test_3():
    print ("This is func_test_3.")




my_list = [func_test_1, func_test_2, func_test_3]
random.choice(my_list)()




"""
for img in image_raw[0:90]:
    X_OS.append(read_img(img))
    if img in file_classZero:
        Y_OS.append(0)
        size0 = size0+1
    elif img in file_classOne:
        X_OS.append(flip1(img))
        X_OS.append(noisy(img))
        #X_OS.append(noisy(img))
        Y_OS.append(1)
        Y_OS.append(1)
        Y_OS.append(1)
        #Y_OS.append(1)
        size1 = size1+3
    else:
        break
"""

#labels = 'IDC (-) ' + str(size0) , 'IDC (+) ' + str(size1)
#sizes = [size0, size1]
#colors =['green', 'red']

#print(len(file_classZero))

#print(len(file_classOne)*2)

plt.figure()
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=30)
plt.axis('equal')




#plt.show()

#while len[]


print('Hello World')
print(cv2.__version__)