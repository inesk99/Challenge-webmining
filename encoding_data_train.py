# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 23:37:57 2022

@author: micka
"""

import cv2 
import face_recognition
from imutils import paths 
import os 
import pickle


my_path = "C:/Users/micka/Downloads/challenge_ines/Photo_Sise_2"

#get paths of each file in folder named Images
#Images here contains my data(folders of various persons)
imagePaths = list(paths.list_images(my_path))
knownEncodings = []
knownNames = []
# loop over the image paths
for imagePath in imagePaths:
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[1].split('.')[0]
    
    print(name)
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb,model='hog')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
#save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "names": knownNames}
#use pickle to save data into a file for later use

f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()


