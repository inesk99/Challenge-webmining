#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import cv2
import numpy as np
import os
import glob

import os 
import pickle

os.chdir("C:/Users/efrintz/Documents/Projet_WM/Challenge-webmining")


# # Import images

# In[2]:
"""

faces_encodings = []

faces_names = []

list_of_files = [f for f in glob.glob("./Photo_Sise_2\\"+'*.jpeg')]

number_files = len(list_of_files)

names = list_of_files.copy()


# # Train the faces

# In[4]:


for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    faces_encodings.append(globals()['image_encoding_{}'.format(i)])
    # Create array of known names
    names[i] = names[i].replace("./Photo_Sise_2\\", "")
    names[i] = names[i].replace(".jpeg","")
    faces_names.append(names[i])
"""

# In[5]:


data = pickle.loads(open('face_encoding','rb').read())

faces_encodings = data["encodings"]
faces_names = data["names"]

# # Emotion

# In[6]:


#import tensorflow
#from keras import models
#model = models.load_model("./model_v6_23.hdf5")


# # Face Recognition

# In[7]:


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# # TEST JUSTE LES NOMS

# In[8]:


#video_capture = cv2.VideoCapture("video.mp4")
cap = cv2.VideoCapture(0)

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'  #getting a haarcascade xml file
face_cascade = cv2.CascadeClassifier()  #processing it for our project
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  #adding a fallback event
    print("Error loading xml file")

cont = 0

while True:
    
    #if cont%2 == 0:
    
    ret, frame = cap.read() # lance vidéo 

    #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # redimensionne 

    rgb_small_frame = frame[:, :, ::-1] # passe en gris 
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if process_this_frame: # si vrai on rentre 


        #faces_locations = face_cascade.detectMultiScale(gray, 1.32, 5) # permet de détecter tout les têtes
        
        face_locations = face_recognition.face_locations(rgb_small_frame) # 

        face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)

        face_names = []


        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces (faces_encodings, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(faces_encodings, face_encoding)

            #face_match_percentage = (1-face_distances)*100

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:

                name = faces_names[best_match_index]

            face_names.append(name)

    print(face_names)
    print(face_locations)
    process_this_frame = not process_this_frame # mets a faux pour sortir

    # Display the results
    #    x     y      w       h 
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 1

        right *= 1

        bottom *= 1

        left *= 1# Draw a rectangle around the face
        
        #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        print(top, right,bottom,left)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)# Input text label with a name below the face

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)# Display the resulting image

    cont += 1
            
    cv2.imshow('Video', frame)# Hit 'q' on the keyboard to quit!

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()

# # TEST OK 

# In[5]:





