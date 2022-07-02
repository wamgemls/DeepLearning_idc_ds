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
sample_size = 1500
test_fac = 0.2

c0_size = 800
c1_size = 300


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
    image = image.reshape(-1)
    #image = image.reshape(50,50,3)
    return image

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
    gaussian = np.random.normal(-7, 10, (50,50,3))
    augmented_img = (img + gaussian).astype(np.uint8)
    return augmented_img

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

def balance_imblearn(DS_img, DS_lbl):
   
    imb = RandomOverSampler(sampling_strategy='auto')
    #ros = RandomUnderSampler(sampling_strategy='auto')
    DS_img, DS_lbl = imb.fit_sample(DS_img, DS_lbl)
    
    return DS_img, DS_lbl


#Dataset Preparation

halfsample=int(sample_size/2) # TwoClasses -> 50/50 dataset

def filelist2imglist(filelist):
    DS_img = []
    for file in filelist:
        img=giveImg(file)
        img=preprocImg(img)
        DS_img.append(img)
    
    return DS_img

def datasetgen(classZero, classOne):
    
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

    return DS_img, DS_lbl

def postprocDS(DS_img):

    DS_img_rs = []

    for image in DS_img:
        image=image.reshape(50,50,3)
        DS_img_rs.append(image)
        
    DS_img_rs = np.array(DS_img_rs)

    return DS_img_rs
    

def oneHotencode(Y_train,Y_test):
    Y_trainHot = to_categorical(Y_train, num_classes = 2)
    Y_testHot = to_categorical(Y_test, num_classes = 2)

    return Y_trainHot, Y_testHot


#main

#enough sample (no over/undersampling needed)
if halfsample <= len(file_classZero) and halfsample <= len(file_classOne):
    
    print('Enough Balanced Data - Strategy A')

    img_classZero = filelist2imglist(file_classZero[0:halfsample])
    img_classOne = filelist2imglist(file_classOne[0:halfsample])

    DS_img, DS_lbl = datasetgen(img_classZero, img_classOne)
    DS_img = postprocDS(DS_img)

    X_train, X_test, Y_train, Y_test = train_test_split(DS_img,DS_lbl,test_size=test_fac)

    Y_trainHot, Y_testHot = oneHotencode(Y_train,Y_test)

#classOne minor (oversampling ClassOne)
elif halfsample <= len(file_classZero) and halfsample > len(file_classOne):
        
    img_classZero = filelist2imglist(file_classZero[0:halfsample])
    img_classOne = filelist2imglist(file_classOne)
    
    if (balancing_strategy==0):
        print('Unbalanced Data => Manual Oversampling ClassOne: Strategy B0')

        os_data = manualoversamp(file_classOne,file_classZero[0:halfsample])
        img_classOne.extend(os_data)
        DS_img,DS_lbl = datasetgen(img_classZero, img_classOne)
        DS_img = postprocDS(DS_img)

        X_train, X_test, Y_train, Y_test = train_test_split(DS_img,DS_lbl,test_size=test_fac)


    if (balancing_strategy==1):
        print('Unbalanced Data => IMB Oversampling ClassOne: Strategy B1')

        DS_img, DS_lbl = datasetgen(img_classZero, img_classOne)
        DS_img, DS_lbl = balance_imblearn(DS_img, DS_lbl)
        DS_img = postprocDS(DS_img)
                
        X_train, X_test, Y_train, Y_test = train_test_split(DS_img,DS_lbl,test_size=test_fac)


    Y_trainHot, Y_testHot = oneHotencode(Y_train,Y_test)

#classZero minor (oversampling classZero)
elif halfsample > len(file_classZero) and halfsample <= len(file_classOne):
    
    img_classZero = filelist2imglist(file_classZero)
    img_classOne = filelist2imglist(file_classOne[0:halfsample])
    
   
    if (balancing_strategy==0):
        print('Unbalanced Data => IMB Oversampling ClassZero: Strategy C0')

        os_data = manualoversamp(file_classZero,file_classOne[0:halfsample])
        img_classZero.extend(os_data)
        DS_img,DS_lbl = datasetgen(img_classZero, img_classOne)
        DS_img = postprocDS(DS_img)

        X_train, X_test, Y_train, Y_test = train_test_split(DS_img,DS_lbl,test_size=test_fac)

    if (balancing_strategy==1):
        print('Unbalanced Data => IMB Oversampling ClassZero: Strategy C1')

        DS_img, DS_lbl = datasetgen(img_classZero, img_classOne)
        DS_img, DS_lbl = balance_imblearn(DS_img, DS_lbl)
        DS_img = postprocDS(DS_img)
                
        X_train, X_test, Y_train, Y_test = train_test_split(DS_img,DS_lbl,test_size=test_fac)

    Y_trainHot, Y_testHot = oneHotencode(Y_train,Y_test)

#not enough sample data
else:
    
    print('not enough data')


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_trainHot.shape)
print(Y_test.shape)
print(Y_testHot.shape)

#final data inspection

filePie(len(file_classZero), len(file_classOne),'Data: Raw')
filePie(np.sum(Y_train)+np.sum(Y_test), (len(Y_train)+len(Y_test))-(np.sum(Y_train)+np.sum(Y_test)),'Data: Processed')

plotcompareImg(giveImg(random.choice(file_classZero)),'IDC (-)',giveImg(random.choice(file_classOne)),'IDC (+)')

plt.show()