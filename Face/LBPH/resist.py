import cv2
import time
import threading
import multiprocessing
import queue
import os
import numpy as np
import pickle
import sys
import system_logging as log
from datetime import datetime
import psutil

dataset = "dataset"
pickles = os.path.join("model", "alpha.resist")
core = os.path.join("model", "core.resist")
protoPath = os.path.join("model", "deploy.prototxt")
modelPath = os.path.join("model", "face_detector.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
recognizer_model = cv2.face.LBPHFaceRecognizer_create()

class Resist:
    def __init__(self):
        try:
            recognizer_model.read(core)
        except:
            self.create_model()
    
    def check_core(self):
        try:
            if not os.path.isfile(core):
                create_model()
            recognizer_model.read(core)
        except:
            pass
    
    def create_model(self):
        self.prepare_training_dataset("dataset")
        file = pickle.load(open(pickles, "rb"))
        label = file["labels"]  # DECENTRIALIZE DICTIONARY
        face = file["faces"]
        recognizer_model.train(face, np.array(label))
        recognizer_model.write(core)

    def update_model(self):
        file = pickle.load(open(pickles, "rb"))
        label = file["labels"]  # DECENTRIALIZE DICTIONARY
        face = file["faces"]
        recognizer_model.read(core)
        recognizer_model.update(face, np.array(label))
        recognizer_model.write(core)

    def adjust_brightness(self,frame):
        avg = frame.mean(axis=0).mean(axis=0)[0]
        avg = np.mean(frame)
        if avg < 100:
            frame = self.adjust_gamma(frame, avg/30)
            return frame

    def adjust_gamma(self,image, gamma):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def draw_rectangle(self,img, rect):
        (startX, startY, endX, endY) = rect.astype("int")
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    def draw_text(self,img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    
    def createCLAHE(self,frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
        res = clahe.apply(frame)
        return res
    
    
    def new_save(self,faces,labels):
        if not os.path.isfile(pickles):  # WRITE FILE IF RESIST FILE DOESN'T EXIST
            data = {"faces": faces, "labels": labels}
            file = open(pickles, "wb")
            file.write(pickle.dumps(data))
            file.close()

    def save_existing(self,newfaces,newlabels):
        if os.path.isfile(pickles):
            faces = []
            labels = []
            file = pickle.load(open(pickles, "rb"))  # LOAD EXISTING RESIST FILE
            label = file["labels"]  # DECENTRIALIZE DICTIONARY
            face = file["faces"]
            for la,fa in zip(label,face):  # APPENDING EACH LIST
                labels.append(la)
                faces.append(fa)
            for la,fa in zip(newlabels,newfaces):
                labels.append(la)
                faces.append(fa)
            data = {"faces": faces, "labels": labels}
            files = open(pickles, "wb")
            files.write(pickle.dumps(data))
            files.close()
            file = pickle.load(open(pickles, "rb"))
    
    def Normalize(self,frame):
        try:
            h,w,c = frame.shape
            frames = []
            boxes = []
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (
                104.0, 177.0, 123.0), swapRB=False, crop=False)  # CONVERT FRAME INTO BLOB FOR DNN INPUT
            detector.setInput(imageBlob)
            detections = detector.forward()
            for i in range(0, detections.shape[2]):  # ITERATE ALL DETECTED FACE
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    (startX, startY, endX, endY) = box.astype("int")
                    gray = self.createCLAHE(frame)
                    equalized = cv2.resize(gray[startY:endY, startX:endX], (400, 400), interpolation=cv2.INTER_LANCZOS4)
                    frames.append(equalized)
                    boxes.append(box.astype("int"))
            return frames, boxes    ##DEFAULT EXPECTED RETURNED VALUE
        except:
            return None, None

    def prepare_training_dataset(self,dataset):
        faces = []
        labels = []
        dirs = os.listdir(dataset)
        for dir_name in dirs:
            if not dir_name.startswith("T"):
                path, dirs, file_name = next(os.walk(os.path.join(dataset, dir_name)))
                total = 0
                label = int(dir_name)
                fullpath = os.path.join(dataset, dir_name)
                for images in file_name:
                    frame = cv2.imread(os.path.join(fullpath, images))
                    h, w, c = frame.shape
                    total += 1
                    sys.stdout.write(
                        "\r [ SYSTEM ] : Current id '{}' - preparing {}/{} Images".format(label, total, len(file_name)))
                    face2, boxes = self.Normalize(frame)
                    for face, box in zip(face2, boxes):
                        if face2[0] is not None:
                            faces.append(face)  # !!DEFAULT EXPECTED RETURN VALUE
                            labels.append(label)  # !!DEFAULT EXPECTED RETURN VALUE
                sys.stdout.write("\n")
                sys.stdout.flush()
                # APPEND DATASET DIR FILE NAME WITH T MEANING ITS BEEN TRAINED
                os.rename(os.path.join(dataset, dir_name),
                        os.path.join(dataset, "T"+dir_name))
        if not os.path.isfile(pickles):
            self.new_save(faces,labels)
        if os.path.isfile(pickles):
            self.save_existing(faces,labels)
    
    def predict(self,face,box,frame):
        (startX, startY, endX, endY) = box.astype("int")
        label, confidence = recognizer_model.predict(face)
        y = startY - 10 if startY - 10 > 10 else startY + 10 
        self.draw_rectangle(frame,box)
        if confidence < 45:
            self.draw_text(frame,str(label)+" Accu: "+str(100-int(confidence)+15),startX, y)
        return label, confidence,frame
        
vid = cv2.VideoCapture(0)
# vid.set(cv2.CAP_PROP_FPS, 60)
Resist = Resist()
Resist.check_core()
while True:
    _,frame = vid.read()
    start = time.time()
    frame = Resist.adjust_brightness(frame)
    faces,boxes = Resist.Normalize(frame)
    if faces != None:
        for face,box in zip(faces,boxes):
            (startX, startY, endX, endY) = box.astype("int")
            y = startY - 10 if startY - 10 > 10 else startY + 10 
            label,confidence,frame = Resist.predict(face,box,frame)
    try:
        frm = "FPS : "+ str(int(1.0 / (time.time() - start)))
        cpu = "CPU : "+str(psutil.cpu_percent()) +"%"
        mem = "MEM : " +str(psutil.virtual_memory()[2])+"%"
        Resist.draw_text(frame,frm,0,100)
        Resist.draw_text(frame,cpu,0,120)
        Resist.draw_text(frame,mem,0,140)
        cv2.imshow("Frame",frame)
    except:
        pass
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break