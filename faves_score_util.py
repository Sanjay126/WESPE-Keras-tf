
import numpy as np
import os
import sys
import cv2
import re
import statistics
import pickle

DATADIR = "./flickr"


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

if __name__=='__main__':
    patches,labels=dataset_creation()
    with open('faves_dataset.pkl','wb') as file:
        pickle.dump((patches,labels),file)

