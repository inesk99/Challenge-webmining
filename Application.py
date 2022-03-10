# Importation des librairie et d'un module model.py
import cv2 as cv
import argparse
import pickle
import face_recognition
import numpy as np
import streamlit as st
from model import ERModel

# Fonction : Récupérer les coordonnées des visages
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    return frameOpencvDnn, bboxes

st.title('Reconnaissance faciale et identification sur des vidéos WEBCAM')


FRAME_WINDOW = st.image([])

start = st.button("Allumer la caméra")
stop = st.button("Eteindre la caméra")

if start : 
    ####################################################
    # Chargement des modèles pré-entrainés
    model = ERModel("model.json", "model_weights.h5")
    data = pickle.loads(open('face_encoding','rb').read())
    
    faces_encodings = data["encodings"]
    faces_names = data["names"]
    
    parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument("--device", default="gpu", help="Device to inference on")
    
    args = parser.parse_args()
    
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    
    ageNet = cv.dnn.readNet(ageProto,ageModel)
    genderNet = cv.dnn.readNet(genderProto,genderModel)
    faceNet = cv.dnn.readNet(faceProto,faceModel)
    
    if args.device == "cpu":
        ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
        
    elif args.device == "gpu":
        ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    
        genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    
        genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")
    
    # Ouvrir la webcam
    cap = cv.VideoCapture(args.input if args.input else 0)
    padding = 20
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    out = cv.VideoWriter('video_output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 5, (frame_width,frame_height))
    
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        frame = cv.flip(frame,1)
        
        if not hasFrame:
            cv.waitKey()
            break
        
        rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frameFace, bboxes = getFaceBox(faceNet, frame)
        
        faces = face_recognition.face_locations(gray)
        
        #if not bboxes:
        #    print("No face Detected, Checking next frame")
        
        face_names = []
        
        for bbox in faces:
            face=frame[max(bbox[0]-padding,0):min(bbox[2]+padding,frame.shape[0]),max(0,bbox[3]-padding):min(bbox[1]+padding,frame.shape[1])]

            ########################################################
            # Détection des visages 
            
            encodings = face_recognition.face_encodings(rgb,[bbox])
            matches = face_recognition.compare_faces (data["encodings"], encodings[0],tolerance=0.6)
            
            name = "Unknown"
            face_distances = face_recognition.face_distance(data["encodings"], encodings[0])
            
            best_match_index = np.argmin(face_distances)
            
            #print("best index matche : {}".format(best_match_index))
    
            if matches[best_match_index]:
                name = faces_names[best_match_index]
    
            face_names.append(name)
            
            #print("Reconnaissance des vissage:  {}".format(face_names))
            
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            ####################################################
            # Prédiction genre
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ####################################################
            # Prédiction age
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            ####################################################
            # Prédiction sentiments
            roi_gray = gray[max(bbox[0]-padding,0):min(bbox[2]+padding,frame.shape[0]),max(0,bbox[3]-padding):min(bbox[1]+padding,frame.shape[1])]
            cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray, (48, 48)), -1), 0)

            pred = model.predict_emotion(cropped_img)
    
    
            label = "{},{},{},{}".format(gender, age,name,pred)
            cv.rectangle(frameFace, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 0), int(round(frameFace.shape[0]/150)), 8)

            cv.rectangle(frameFace, (bbox[1], bbox[0] - 35), (bbox[1], bbox[2]), (255, 255, 255), cv.FILLED)
            
            font = cv.FONT_HERSHEY_DUPLEX

            k = 0
            
            for i, line in enumerate(label.split(',')):
                k = k+25
                
                try:
                    if line in ["Angry", "Disgust","Fear", "Happy","Neutral", "Sad","Surprise"]:
                        cv.putText(frameFace, line, (bbox[1] + 6,  bbox[0] - 6 + k), font, 1.0, (0, 255, 255), 1)
                    elif line.startswith("("):
                        cv.putText(frameFace, line+" "+str(round(max(agePreds[0].tolist())*100,2)), (bbox[1] + 6,  bbox[0] - 6 + k), font, 1.0, (255, 255, 0), 1)
                    elif line == "Male":
                        cv.putText(frameFace, line+" "+str(round(max(genderPreds[genderPreds[0].argmax()].tolist())*100,2)), (bbox[1] + 6,  bbox[0] - 6 + k), font, 1.0, (255, 0, 0), 1)
                    elif line == "Female":
                        cv.putText(frameFace, line, (bbox[1] + 6,  bbox[0] - 6 + k), font, 1.0, (0, 0, 255), 1)
                    else:
                        cv.putText(frameFace, line, (bbox[1] + 6,  bbox[0] - 6 + k), font, 1.0, (255, 0, 255), 1)
                except:
                    pass
                
        frame_ok = cv.cvtColor(frameFace, cv.COLOR_RGB2BGR)
    
        # Affichage de l'image sur l'application
        FRAME_WINDOW.image(frame_ok)
        
        out.write(cv.cvtColor(frame_ok, cv.COLOR_RGB2BGR))
    
        if stop :
            break 
            cap.release()
            out.release()

cv.destroyAllWindows()
