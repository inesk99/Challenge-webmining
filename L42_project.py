# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:10:41 2022

@author: efrintz
"""

import cv2
print(cv2.__version__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture("./video_test.mp4")

while True:
    ret, frame = cap.read()
    tickmark = cv2.getTickCount()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=0)
    for x, y, w, h in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    if cv2.waitKey(1)==ord('q'):
        break
    fps = cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
    cv2.imshow('video', frame)

cap.release()
cv2.destroyAllWindows()




