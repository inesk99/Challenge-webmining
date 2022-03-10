# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:49:34 2022

@author: Inès
"""

import cv2
import numpy as np
import streamlit as st
#from deepface import DeepFace
#import numpy as np
#from keras.preprocessing import image

#import time

import face_recognition
#import os
#import glob

import pickle

st.title('Application reconnaissance facile par Webcam')


FRAME_WINDOW = st.image([])

start = st.button("Allumer la caméra")
stop = st.button("Eteindre la caméra")

if start : 
     
    cam = cv2.VideoCapture(0)

    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))

    #frame_width = int(cam.get(3))
    #frame_height = int(cam.get(4))

    out = cv2.VideoWriter('output_4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))
    
    ###########################################################################
    # Charger le modèle pré-entrainé
    data = pickle.loads(open('face_encoding','rb').read())
    faces_encodings = data["encodings"]
    faces_names = data["names"]
    
    # Initialisation 
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    cont = 0
    
    face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'  # Fichier xml 
    face_cascade = cv2.CascadeClassifier()  #processing it for our project
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  #adding a fallback event
        print("Error loading xml file")
    
    ###########################################################################

    while True : 
        
        ret, frame = cam.read()
        
        frame = cv2.flip(frame,1)
        
        rgb_small_frame = frame[:, :, ::-1] # Passe en gris
        
        if process_this_frame: # si vrai on rentre 
        
            face_locations = face_recognition.face_locations(rgb_small_frame) 
            face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
            face_names = []


            for face_encoding in face_encodings:

                matches = face_recognition.compare_faces (faces_encodings, face_encoding)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(faces_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:

                    name = faces_names[best_match_index]

                face_names.append(name)

        ###print(face_names)
        ###print(face_locations)
        process_this_frame = not process_this_frame # Mets a faux pour sortir

        # Display the results
        #    x     y      w       h 
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            
            top *= 1
            right *= 1
            bottom *= 1
            left *= 1 # Draw a rectangle around the face
            
            #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            ###print(top, right,bottom,left)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)# Input text label with a name below the face

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX

            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)# Display the resulting image

        cont += 1
        
        cv2.imshow('Video', frame)
        
        frame_col = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        frame_col = cv2.flip(frame_col,1)

        out.write(cv2.flip(cv2.cvtColor(frame_col, cv2.COLOR_RGB2BGR),1))
        
        # Affichage de l'image sur l'application
        FRAME_WINDOW.image(cv2.flip(frame_col,1))
        
        
        if stop : 
            break 
            cam.release()
            out.release()

        
        #######################################################################
        
        #ret, frame = cam.read()
        
        #frame_col = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        #frame_col = cv2.flip(frame_col,1)

        #out.write(cv2.cvtColor(frame_col, cv2.COLOR_RGB2BGR))

        
        #FRAME_WINDOW.image(frame_col)
        
        #if stop : 
        #    break 
        #    cam.release()
        #    out.release()
        

   
#cv2.destroyAllWindows()

