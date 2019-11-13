import numpy as np
import os
import cv2
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import re2

############################################################################################


DATADIR = "/home/satyam/Desktop/personal/flickr"

# while true:  
#   path = os.path.join(DATADIR,faves_score) 
#    for img in os.listdir(path):  
#        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
#        plt.imshow(img_array, cmap='gray')


i=1
for file_name in DATADIR:
    
    res = re2.findall("flickr_"+'i' + "_(\d+).jpg", file_name)
    if not res: 
        continue

    i++
    faves_score = res[0]



############################################################################################

# RESISIZING THE IMAGE

img = cv2.imread('your_image.jpg')
res = cv2.resize(img, dsize=(720, 1280), interpolation=cv2.INTER_CUBIC)

#############################################################################################

median = np.median(faves)

for i in (len):

	if FFS> median:
		quality = "high"

	else:
		quality = "low"

	df['QUALITY'] = quality    #adding extra column


##############################################################################################

img_patch = load_sample_image("image to be patched")
patches = image.extract_patches_2d(img_patch, (224, 224))


##### VGG-19 style ######


def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Activation('softmax'))


