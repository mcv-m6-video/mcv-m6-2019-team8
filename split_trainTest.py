import random
import os
import subprocess
import sys
import cv2
from os import walk
import glob

def splitA(image_dir):
    ind = 0
    n_elem=len(glob.glob(image_dir+'/*.jpg'))
    n_train = round(n_elem*0.25)
    n_val = n_elem-n_train

    path, dirs, files = next(os.walk(image_dir))
    data_size = len(files)

    data_test_size = int(0.25 * data_size)
    test_array = random.sample(range(data_size), k=data_test_size)
  

    for f in os.listdir(image_dir):
        if(f.split(".")[1] == "jpg"):
            ind += 1   

            directorio = image_dir+'/train/'
            try:
              os.stat(directorio)
            except:
              os.mkdir(directorio)
            directorio = image_dir+'/test/'
            try:
              os.stat(directorio)
            except:
              os.mkdir(directorio)

            if ind <=n_train:
               img = cv2.imread(image_dir+'/'+f,3)
               cv2.imwrite(image_dir+'/train/'+f,img)
            else:
               img = cv2.imread(image_dir+'/'+f,3)
               cv2.imwrite(image_dir+'/test/'+f,img)
def splitB(image_dir):

    path, dirs, files = next(os.walk(image_dir))
    data_size = len(files)

    ind = 0
    data_test_size = int(0.25 * data_size)
    test_array = random.sample(range(data_size), k=data_test_size)
    
    for f in os.listdir(image_dir):
        if(f.split(".")[1] == "jpg"):
            ind += 1   

            directorio = image_dir+'/train/'
            try:
              os.stat(directorio)
            except:
              os.mkdir(directorio)
            directorio = image_dir+'/test/'
            try:
              os.stat(directorio)
            except:
              os.mkdir(directorio)

            if ind in test_array:
               img = cv2.imread(image_dir+'/'+f,3)
               cv2.imwrite(image_dir+'/train/'+f,img)
            else:
               img = cv2.imread(image_dir+'/'+f,3)
               cv2.imwrite(image_dir+'/test/'+f,img)

def splitCD(image_dir, per):

    path, dirs, files = next(os.walk(image_dir))
    data_size = len(files)

    ind = 0
    data_test_size = int(per * data_size)
    test_array = random.sample(range(data_size), k=data_test_size)
    
    for f in os.listdir(image_dir):
        if(f.split(".")[1] == "jpg"):
            ind += 1   

            directorio = image_dir+'/train/'
            try:
              os.stat(directorio)
            except:
              os.mkdir(directorio)
            directorio = image_dir+'/test/'
            try:
              os.stat(directorio)
            except:
              os.mkdir(directorio)

            if ind in test_array:
               img = cv2.imread(image_dir+'/'+f,3)
               cv2.imwrite(image_dir+'/train/'+f,img)
            else:
               img = cv2.imread(image_dir+'/'+f,3)
               cv2.imwrite(image_dir+'/test/'+f,img)


image_dir= 'pruebas/data/train/tomato'
per = 0.3 # per must be between 0 and 1

#splitA(image_dir)
#splitB(image_dir)
splitCD(image_dir,per)
