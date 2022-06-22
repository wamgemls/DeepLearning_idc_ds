from matplotlib import image
from matplotlib.pyplot import axis
import numpy as np
from glob import glob
import fnmatch
import matplotlib.pylab as plt
import cv2
import random
import tensorflow_datasets as tfds
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

"""
Balancing Strategy:
$0 = Manual Oversampling
$1 = imblearn Package
"""

balancing_strategy = 0
sample_size = 202
test_fac = 0.2

c0_size = 500
c1_size = 200


#Data Aquisition

image_raw = glob('../IDC_regular_ps50_idx5/**/*.png', recursive=True)

patternZero = '*class0.png'
patternOne = '*class1.png'

file_classZero = fnmatch.filter(image_raw, patternZero)
file_classOne = fnmatch.filter(image_raw, patternOne)

file_classZero = file_classZero[0:c0_size]
file_classOne = file_classOne[0:c1_size]

def filePie(size0, size1,title):
    labels = 'IDC (-) ' + str(size0) , 'IDC (+) ' + str(size1)
    sizes = [size0, size1]
    colors =['green', 'red']
    plt.figure()
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=90)
    plt.axis('equal')
    plt.suptitle(title, fontsize=16)

def plotImgFile(file_image):
    image = cv2.imread(file_image)
    image = cv2.resize(image, (50,50))
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def plotcompareImg(imageA,titleA,imageB,titleB): 
    plt.figure()
    plt.subplot(1,2,1)
    plt.title(titleA)
    plt.imshow(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB))
    plt.subplot(1,2,2)
    plt.title(titleB)
    plt.imshow(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB))
    return

#Image Processing 1

def giveImg(file_image):
    image = cv2.imread(file_image)
    image = cv2.resize(image, (50,50))
    return image
    
def preprocImg(image):
    image = image/255.0
    imgFlat = image.reshape(-1)
    return imgFlat

#Image Processing 2

def flip1(file_img):
    img = giveImg(file_img)
    augmented_img = np.fliplr(img)
    return augmented_img

def flip2(file_img):
    img = giveImg(file_img)
    augmented_img = np.flipud(img)
    return augmented_img

def noisy(file_img):
    img = giveImg(file_img)
    noisy_img = np.random.poisson(img / 255.0 * 400) / 400 * 255
    noisy_img = cv2.resize(noisy_img, (50,50))
    noisy_img = noisy_img.astype(np.uint8)
    return noisy_img

augmentlist = [flip1, flip2, noisy]

#Balancing 
#1_Manual

def manualoversamp(file_osclass,file_nclass):
    
    disbalance = len(file_nclass) - len(file_osclass)
    os_images=[]
    
    for i in range(disbalance):

        augfunc = random.choice(augmentlist)
        random_image= random.choice(file_osclass)
        img = augfunc(random_image)
        img = preprocImg(img)
        os_images.append(img)

    return os_images

#2_Imblearn

def balance_imblearn(X_train,X_test,Y_train,Y_test):
   
    imb = RandomOverSampler(sampling_strategy='auto')
    #ros = RandomUnderSampler(sampling_strategy='auto')
    X_train_imb, Y_train_imb = imb.fit_sample(X_train, Y_train)
    X_test_imb, Y_test_imb = imb.fit_sample(X_test, Y_test)

    return X_train_imb, X_test_imb, Y_train_imb, Y_test_imb


#Dataset Preparation

halfsample=int(sample_size/2) # TwoClasses -> 50/50 dataset

def filelist2imglist(filelist):
    DS_img = []
    for file in filelist:
        img=giveImg(file)
        img=preprocImg(img)
        DS_img.append(img)
    
    return DS_img

def datasetgen(classZero, classOne, test_fac):
    
    DS_img = []
    DS_lbl = []
    
    DS_img.extend(classZero)
    DS_img.extend(classOne)

    DS_lbl.extend(np.zeros(len(classZero)))
    DS_lbl.extend(np.ones(len(classOne)))

    DS_img=np.array(DS_img)
    DS_lbl=np.array(DS_lbl)

    #shuffle
    s = np.arange(DS_img.shape[0])
    np.random.shuffle(s)
    DS_img = DS_img[s]
    DS_lbl = DS_lbl[s] 


    X_train, X_test, Y_train, Y_test = train_test_split(DS_img,DS_lbl,test_size=test_fac)

    return X_train, X_test, Y_train, Y_test

def oneHotencode(Y_train,Y_test):
    Y_trainHot = to_categorical(Y_train, num_classes = 2)
    Y_testHot = to_categorical(Y_test, num_classes = 2)

    return Y_trainHot, Y_testHot


#main

if halfsample <= len(file_classZero) and halfsample <= len(file_classOne):
    #enough Sample
    print('Enough Balanced Data - Strategy A')

    img_classZero = filelist2imglist(file_classZero[0:halfsample])
    img_classOne = filelist2imglist(file_classOne[0:halfsample])

    X_train, X_test, Y_train, Y_test = datasetgen(img_classZero, img_classOne, test_fac)

    Y_trainHot, Y_testHot = oneHotencode(Y_train,Y_test)


elif halfsample <= len(file_classZero) and halfsample > len(file_classOne):
        
    img_classZero = filelist2imglist(file_classZero[0:halfsample])
    img_classOne = filelist2imglist(file_classOne)
    
    if (balancing_strategy==0):
        print('Unbalanced Data => Manual Oversampling ClassOne: Strategy B0')

        os_data = manualoversamp(file_classOne,file_classZero[0:halfsample])
        img_classOne.extend(os_data)
        X_train, X_test, Y_train, Y_test = datasetgen(img_classZero, img_classOne, test_fac)

    if (balancing_strategy==1):
        print('Unbalanced Data => IMB Oversampling ClassOne: Strategy B1')

        X_train, X_test, Y_train, Y_test = datasetgen(img_classZero, img_classOne, test_fac)
        X_train, X_test, Y_train, Y_test = balance_imblearn(X_train,X_test,Y_train,Y_test)

    Y_trainHot, Y_testHot = oneHotencode(Y_train,Y_test)


elif halfsample > len(file_classZero) and halfsample <= len(file_classOne):
    
    img_classZero = filelist2imglist(file_classZero)
    img_classOne = filelist2imglist(file_classOne[0:halfsample])
    
   
    if (balancing_strategy==0):
        print('Unbalanced Data => IMB Oversampling ClassZero: Strategy C0')

        os_data = manualoversamp(file_classZero,file_classOne[0:halfsample])
        img_classZero.extend(os_data)
        X_train, X_test, Y_train, Y_test = datasetgen(img_classZero, img_classOne, test_fac)

    if (balancing_strategy==1):
        print('Unbalanced Data => IMB Oversampling ClassZero: Strategy C1')

        X_train, X_test, Y_train, Y_test = datasetgen(img_classZero, img_classOne, test_fac)
        X_train, X_test, Y_train, Y_test = balance_imblearn(X_train,X_test,Y_train,Y_test)

    Y_trainHot, Y_testHot = oneHotencode(Y_train,Y_test)

else:
    #not enough data
    print('not enough data')


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_trainHot.shape)
print(Y_test.shape)
print(Y_testHot.shape)


#Final Data inspection

filePie(len(file_classZero), len(file_classOne),'Data: Raw')
filePie(np.sum(Y_train)+np.sum(Y_test), (len(Y_train)+len(Y_test))-(np.sum(Y_train)+np.sum(Y_test)),'Data: Processed')

plotcompareImg(giveImg(random.choice(file_classZero)),'IDC (-)',giveImg(random.choice(file_classOne)),'IDC (+)')

plt.show()