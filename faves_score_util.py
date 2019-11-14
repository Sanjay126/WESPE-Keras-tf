
import numpy as np
import os
import sys
import cv2
import pandas as pd
import re
import statistics
import pickle

from PIL import Image
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from datagen import DataGenerator as DataGenerator2
import datagen
from tensorflow.keras.applications.vgg_19 import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import copy
from sklearn.feature_extraction import image



DATADIR = "/home/satyam/Desktop/personal/flickr"


def dataset_creation():

    faves_score =[]
    image=[]
    size=len(faves_score)
    patches=np.empty((size,224,224,3))
    labels=np.empty((size,2))
    for file_name in os.listdir(DATADIR):
            
        faves = re.findall(r"_0.(\d+).jpg", file_name)

        if not faves: 
          continue
        faves_score.append(float(faves[0]))                
    
    median = statistics.median(faves_score)
    print (median)
    img_name_list=os.listdir(DATADIR)
    for i in range(len(faves_score)):
        
        img_name=img_name_list[i]
        img_path = os.path.join(DATADIR , img_name)
        img=cv2.imread(img_path)
        resized=cv2.resize(img, (720,1280))
        patch_x = np.random.choice(range(0,496)) 
        patch_y=np.random.choice(range(0,1096))
        patches[i,]=resized[patch_x:patch_x+224][patch_y:patch_y+224]

        if (faves_score[i] > median):
            labels[i,]=np.array([0,1])
        else:
            labels[i,]=np.array([1,0])
    return patches,labels




class VGG_19():

    HEIGHT = 224
    WIDTH = 224

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

    def build_finetune_model(base_model, dropout, fc_layers, num_classes):
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = Flatten()(x)
        for fc in fc_layers:
            # New FC layer, random init
            x = Dense(fc, activation='relu')(x) 
            x = Dropout(dropout)(x)

        # New softmax layer
        predictions = Dense(num_classes, activation='softmax')(x) 
        
        finetune_model = Model(inputs=base_model.input, outputs=predictions)

        return finetune_model

    class_list = ["Original","Tampered"]
    FC_LAYERS = [1024, 1024]
    NUM_EPOCHS = 100
    BATCH_SIZE = 25
    lr=0.00005
    dropout = 0.5

    finetune_model = build_finetune_model(base_model, dropout=dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))

    adam = Adam(lr=0.00001)
    finetune_model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])

    filepath="./checkpoints/" + "MobileNetV2" + "_model_weights2.h5"
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

    checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8,
                                           shuffle=True, callbacks=callbacks_list,validation_data=valid_generator,validation_freq=5,use_multiprocessing=True)


if __name__=='__main__':
    
