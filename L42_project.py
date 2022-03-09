# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:10:41 2022

@author: efrintz
"""

import cv2
print(cv2.__version__)
import operator

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

cap = cv2.VideoCapture("./video_test.mp4")
width = int(cap.get(3))
marge = 70

"""
while True:
    ret, frame = cap.read()
    tickmark = cv2.getTickCount()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    for x, y, w, h in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    if cv2.waitKey(1)==ord('q'):
        break
    fps = cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
    cv2.imshow('video', frame)
"""

while True:
    ret, frame = cap.read()
    tab_face = []
    tickmark = cv2.getTickCount()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Reconnaissance de face
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)
    for x, y, w, h in face:
        tab_face.append([x, y, x+w, y+h])
    # Reconnaissance de profile 1
    face = profile_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)
    for x, y, w, h in face:
        tab_face.append([x, y, x+w, y+h])
    # Reconnaissance de profile 2
    gray2 = cv2.flip(gray, 1)
    face = profile_cascade.detectMultiScale(gray2, scaleFactor=1.2, minNeighbors=2)
    for x, y, w, h in face:
        tab_face.append([x, y, x+w, y+h])
    tab_face = sorted(tab_face, key=operator.itemgetter(0,1))
    index = 0
    for x, y, x2, y2 in tab_face:
        if not index or (x-tab_face[index-1][0]>marge or y-tab_face[index-1][1]>marge):
            cv2.rectangle(frame, (x,y), (x2,y2), (0,0,255), 2)
        index+=1
    if cv2.waitKey(1)==ord('q'):
        break
    fps = cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
    cv2.imshow('video', frame)

cap.release()
cv2.destroyAllWindows()




