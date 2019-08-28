import cv2
import time
import threading
import multiprocessing
import queue
import os
import numpy as np
import pickle
import sys

dataset = "dataset"
pickles = os.path.join("model","alpha.resist")
core = os.path.join("model","core.resist")
recognizer_model = cv2.face.LBPHFaceRecognizer_create()
protoPath = os.path.join("model", "deploy.prototxt")
modelPath = os.path.join("model","res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
recognizer_model.read(core)

def adjust_brightness(frame):
    avg = frame.mean(axis=0).mean(axis=0)
    if avg <100:
        frame = adjust_gamma(frame,avg/60)
        return frame

def adjust_gamma(image, gamma):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def draw_rectangle(img, rect):
    (startX, startY, endX, endY) = rect.astype("int")
    cv2.rectangle(img, (startX, startY), (endX, endY),(0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def normalize_face_data(frame,w,h):
    frames = []
    boxes = []
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False) ## CONVERT FRAME INTO BLOB FOR DNN INPUT
    detector.setInput(imageBlob)
    detections = detector.forward()
    for i in range(0, detections.shape[2]): ## ITERATE ALL DETECTED FACE
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        confidence = detections[0, 0, i, 2]
        # print("{} {}".format(startY,endY),"{} {}".format(startX,endX))
        if confidence > 0.5:
            (startX, startY, endX, endY) = box.astype("int")
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frames.append(gray[startY:endY,startX:endX])
            boxes.append(box.astype("int")) ##!!DEFAULT EXPECTED RETURNED VALUE
    return frames,boxes

def prepare_training_dataset(dataset):
    faces = []
    labels = []
    dirs = os.listdir(dataset)
    for dir_name in dirs:
        if dir_name.startswith("T"):
            continue
        if dir_name.startswith("."):
            continue
        path,dirs,file_name = next(os.walk(os.path.join(dataset,dir_name)))
        total = 0
        label = int(dir_name)
        fullpath = os.path.join(dataset,dir_name)
        # print("Trained data : uid[{}]".format(labels[:1]))
        for images in file_name:
            frame = cv2.imread(os.path.join(fullpath,images))
            h,w,c = frame.shape
            total += 1
            sys.stdout.write("\r [ SYSTEM ] : Current id '{}' - preparing {}/{} Images".format(label,total,len(file_name)))
            face2,boxes = normalize_face_data(frame,w,h)
            for face,box in zip(face2,boxes):
                # cv2.imshow("frame",face)
                # cv2.waitKey(0)
                if face2[0] is not None:
                    faces.append(face)##!!DEFAULT EXPECTED RETURN VALUE
                    labels.append(label)##!!DEFAULT EXPECTED RETURN VALUE
            sys.stdout.flush()
        os.rename(os.path.join(dataset,dir_name),os.path.join(dataset,"T"+dir_name)) ## APPEND DATASET DIR FILE NAME WITH T MEANING ITS BEEN TRAINED        
    if not os.path.isfile(pickles): ## WRITE FILE IF RESIST FILE DOESN'T EXIST
        data = {"faces": faces,"labels":labels}
        file = open(pickles,"wb")
        file.write(pickle.dumps(data))
        file.close()
    else: ## APPEND AND WRITE FILE IF FILE EXIST
        
        file = pickle.load(open(pickles,"rb")) ## LOAD EXISTING RESIST FILE
        label = file["labels"]## DECENTRIALIZE DICTIONARY
        face = file["faces"]

        for la in label: ## APPENDING EACH LIST
            labels.append(la)
        for fa in face:
            faces.append(fa)

        data = {"faces": faces,"labels":labels}
        files = open(pickles,"wb")
        files.write(pickle.dumps(data))
        files.close()
        file = pickle.load(open(pickles,"rb"))
    # return faces,labels
    sys.stdout.write("\n [ SYSTEM ] : Completed")

def create_model():
    prepare_training_dataset("dataset") 
    file = pickle.load(open(pickles,"rb"))
    label = file["labels"]## DECENTRIALIZE DICTIONARY
    face = file["faces"]
    recognizer_model.train(face,np.array(label))
    recognizer_model.write(core)

def update_model():
    file = pickle.load(open(pickles,"rb"))
    label = file["labels"]## DECENTRIALIZE DICTIONARY
    face = file["faces"]
    recognizer_model.read(core)
    recognizer_model.update(face,np.array(label))
    recognizer_model.write(core)

##TODO : DATA NORMALIZATION
def predict(frame):
    label, confidence = recognizer_model.predict(frame)
    return label,confidence
    # labels = []
    # rects = []
    # confidences = []
    # recognizer_model.read(core)
    # faces, boxes = normalize_face_data(frame,w,h) ## CALL DATA NORM FUNCTION
    # for face,box in zip(faces,boxes):
    #     label, confidence = recognizer_model.predict(face)
    #     labels.append(label)
    #     confidences.append(confidence)
    #     rects.append(box)
    #     faces.append(face)
    # return labels,confidences,rects