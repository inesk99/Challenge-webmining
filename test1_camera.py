# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:49:34 2022

@author: Inès
"""

import cv2
import numpy as np
import streamlit as st
 

st.title('Application reconnaissance facile par Webcam')


FRAME_WINDOW = st.image([])



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
        
        out.write(frame_col) 
        
        FRAME_WINDOW.image(frame_col)
        #cv2.imshow("frame", frame)
        
        if stop : 
            break 
            cam.release()
            out.release()

   
#cv2.destroyAllWindows()

