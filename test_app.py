# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:10:00 2022

@author: Inès
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:14:54 2022

@author: Clément
"""

import pandas as pd
import cv2
import os
import streamlit as st
import numpy as np
#os.chdir("C:/Users/cleme/OneDrive/Cours M2 SISE/Challenge reconnaissance faciale")
os.getcwd()
#Titre de l'application
st.title('Application reconnaissance facile par Webcam')

st.write('')
#st.sidebar.header('Barre de paramètre')


#Initialiser l'enregistrement
cam = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30,(frame_width,frame_height))


start = st.button("Allumer la caméra")
stop = st.button("Eteindre la caméra")


#Bouton interface
while start:
    ret, frame = cam.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    out.write(frame)
    
    FRAME_WINDOW.image(frame)
    
    if stop == True:
        break

record = st.button("Exporter la video")


cam.release()    
out.release()



#Fenetre de la webcam
#camera = st.camera_input("Take a picture")