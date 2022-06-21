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
sample_size = 700
test_fac = 0.2

syn_DS_size_c0 = 350
syn_DS_size_c1 = 500


if (sample_size % 2):
        sample_size = sample_size-1

halfsample=int(sample_size/2) # TwoClasses -> 50/50 dataset


#Data Aquisition

image_raw = glob('../IDC_regular_ps50_idx5/**/*.png', recursive=True)

for filename in image_raw[0:10]:
    print(filename)

patternZero = '*class0.png'
patternOne = '*class1.png'

file_classZero = fnmatch.filter(image_raw, patternZero)
file_classOne = fnmatch.filter(image_raw, patternOne)

print("IDC (-)\n\n", file_classZero[0:5], '\n')
print("IDC (+)\n\n", file_classOne[0:5], '\n')

file_classZero = file_classZero[0:syn_DS_size_c0]
file_classOne = file_classOne[0:syn_DS_size_c1]

def filePie(file_classZero, file_classOne):
    labels = 'IDC (-) ' + str(len(file_classZero)) , 'IDC (+) ' + str(len(file_classOne))
    sizes = [len(file_classZero), len(file_classOne)]
    colors =['lightskyblue', 'red']
    plt.figure()
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=30)
    plt.axis('equal')



#Image Processing 1

def read_img(file_image):
    img = cv2.imread(file_image)
    img = cv2.resize(img, (50,50))
    return img

def fileplotImg(file_image):
    image = cv2.imread(file_image)
    image = cv2.resize(image, (50,50))
    
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def plotImg(image):
    image = cv2.resize(image, (50,50))
    
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


#Image Processing 2

def flip1(file_img):
    img = read_img(file_img)
    augmented_img = np.fliplr(img)
    return augmented_img

def flip2(file_img):
    img = read_img(file_img)
    augmented_img = np.flipud(img)
    return augmented_img

def noisy(file_img):
    img = read_img(file_img)
    noisy = np.random.poisson(img / 255.0 * 400) / 400 * 255
    noisy = cv2.resize(noisy, (50,50))
    noisy = noisy.astype(np.uint8)
    return noisy

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
        os_images.append(img)

    return os_images

#2_Imblearn

def balance_imblearn(X_train,X_test,Y_train,Y_test):
   
    X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
    X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
    X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
    X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)

    ros = RandomOverSampler(sampling_strategy='auto')
    #ros = RandomUnderSampler(sampling_strategy='auto')
    X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
    X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)

    return X_trainRos, X_testRos, Y_trainRos, Y_testRos


#Dataset Preparation

def file2img(filelist):
    DS_img = []
    for file in filelist:
        DS_img.append(read_img(file))
    
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






random_image0= random.choice(file_classZero)
random_image1= random.choice(file_classOne)

fileplotImg(random_image0)
fileplotImg(random_image1)



#main

if halfsample <= len(file_classZero) and halfsample <= len(file_classOne):
    #enough Sample
    print('Enough Balanced Data - Strategy A')

    file_classZero = file_classZero[0:halfsample]
    file_classOne = file_classOne[0:halfsample]

    img_classZero = file2img(file_classZero)
    img_classOne = file2img(file_classOne)
    
    X_train, X_test, Y_train, Y_test = datasetgen(img_classZero, img_classOne, test_fac)

    Y_trainHot, Y_testHot = oneHotencode(Y_train,Y_test)


elif halfsample <= len(file_classZero) and halfsample > len(file_classOne):
        
    img_classZero = file2img(file_classZero[0:halfsample])
    img_classOne = file2img(file_classOne)
    
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
    
    img_classZero = file2img(file_classZero)
    img_classOne = file2img(file_classOne[0:halfsample])
    
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



#plt.show()

print('Hello World')
print(cv2.__version__)