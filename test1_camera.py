# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:49:34 2022

@author: Inès
"""

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
import numpy as np
from keras.preprocessing import image

 

st.title('Application reconnaissance facile par Webcam')


FRAME_WINDOW = st.image([])

##################################
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
##################################


start = st.button("Allumer la caméra")
stop = st.button("Eteindre la caméra")

if start : 
     
    cam = cv2.VideoCapture(0)

    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))

    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))


    out = cv2.VideoWriter('output_4.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30,(frame_width,frame_height))

    while True : 
    
    
        ret, frame = cam.read()
        
        frame_col = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_col = cv2.flip(frame_col,1)
    ##################################
        
    face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  #getting a haarcascade xml file
    face_cascade = cv2.CascadeClassifier()  #processing it for our project
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  #adding a fallback event
        print("Error loading xml file")
    
    cont = 0
    
    while True:
        
        #if cont%2 == 0:
        
        ret, frame = video_capture.read()
    
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
        rgb_small_frame = small_frame[:, :, ::-1]
    
    
        if process_this_frame:
    
    
            faces_locations = face_haar_cascade.detectMultiScale(rgb_small_frame, 1.32, 5)
            
            #face_locations = face_recognition.face_locations( rgb_small_frame)
    
            face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
    
            face_names = []
    
    
            for face_encoding in face_encodings:
    
                matches = face_recognition.compare_faces (faces_encodings, face_encoding)
    
                name = "Unknown"
    
                face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
    
                #face_match_percentage = (1-face_distances)*100
    
                best_match_index = np.argmin(face_distances)
    
                if matches[best_match_index]:
    
                    name = faces_names[best_match_index]
    
                    face_names.append(name)
    
    
        process_this_frame = not process_this_frame
    
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            
           
    
            top *= 4
    
            right *= 4
    
            bottom *= 4
    
            left *= 4# Draw a rectangle around the face
    
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)# Input text label with a name below the face
    
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    
            font = cv2.FONT_HERSHEY_DUPLEX
    
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)# Display the resulting image
    
        cont += 1
                
        cv2.imshow('Video', frame)# Hit 'q' on the keyboard to quit!
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        
        ##################################
        
        out.write(frame_col) 
        
        FRAME_WINDOW.image(frame_col)
        #cv2.imshow("frame", frame)
        
        if stop : 
            break 
            cam.release()
            out.release()

   
#cv2.destroyAllWindows()

