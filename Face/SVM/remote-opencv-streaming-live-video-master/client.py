import cv2
import numpy as np
import socket
import sys
import pickle
import struct
from io import StringIO
import json
import numpy as np
# import argparse
import imutils
import time
import cv2
import os
import PySimpleGUI as sg
from imutils.face_utils import FaceAligner
from google_speech import Speech
import system_logging as log
import numpy as np
import argparse
import imutils
import pickle
import queue
import threading
import json
import time
import send
import dlib
import cv2
import os

cap=cv2.VideoCapture(0)
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost',8089))
protoPath = os.path.join("face_detection_model", "deploy.prototxt")
modelPath = os.path.join("face_detection_model",
    "res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
unlocked = False
thumbnail_created = False
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())
passes = 0
step = 0
datas = queue.Queue(maxsize=0)
temp = queue.Queue(maxsize=0)
attended = {"user_id":""}
sent = {"sent":False}

def convert_json(uid,cam_id,photo):
    dicts = json.dumps(dict)
    return dicts

def adjust_gamma(image):
    frame = image
    avg = np.average(frame)
    if avg < 130:
        invGamma = 1.0 / (avg/35)
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        frame = cv2.LUT(frame, table)
    return frame

def hello(name):
    sox_effects = ("speed", "1.1")
    speech = Speech("Hey there! {}!, Welcome to 10th Vocational Highschools!".format(name), "en")
    speech.play(sox_effects)


def sends():
    while datas.not_empty:
        if temp.not_empty:
            json = datas.get()
            icon = temp.get()
            if json["user_id"] in attended["user_id"]:
                continue
            attended.update({"user_id":json["user_id"]})
            # while not sent["sent"]:
            send.post_attendance(json,icon)

def createCLAHE(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
        res = clahe.apply(frame)
        return res

if __name__ == "__main__":
    t = threading.Thread(target=sends)
    t.daemon = True
    t.start()
    vid = cv2.VideoCapture(0)
    out = cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc(*"MJPG"), 30,(int(vid.get(3)),int(vid.get(4))))
    while unlocked != True:
        _,image =vid.read()
        if not _:
            log.err("Camera Disconnected or not detected")
            break
        original = image.copy()
        image = adjust_gamma(image)
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fH <80:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),(0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
                # cv2.rectangle(original, (startX, startY), (endX, endY),(0, 0, 255), 2)
                copied = original.copy()
                cropped = copied[startY:endY,startX:endX]
                cv2.imwrite("temp.png",cropped)
                cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                # cv2.putText(original, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                if (proba*100) >= 93 and name != "Guest":
                    unlocked = False
                    dict = {"user_id":name,"camera_id":"5d4c276616171e2938004c72"}
                    file = {"photo":open('temp.png', 'rb')}
                    temp.put(file)
                    datas.put(dict)
        cv2.imshow("frame",image)
        memfile = StringIO.StringIO()
        np.save(memfile, image)
        memfile.seek(0)
        data = json.dumps(memfile.read().decode('latin-1'))

        clientsocket.sendall(struct.pack("L", len(data))+data)
        out.write(image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):                
            break
    out.release()
    vid.release()
    cv2.destroyAllWindows()

while(cap.isOpened()):
  ret,frame=cap.read()

  memfile = StringIO.StringIO()
  np.save(memfile, frame)
  memfile.seek(0)
  data = json.dumps(memfile.read().decode('latin-1'))

  clientsocket.sendall(struct.pack("L", len(data))+data)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()