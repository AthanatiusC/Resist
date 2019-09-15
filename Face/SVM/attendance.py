# YOLO object detection using a webcam
# Exact same demo as the read from disk, but instead of disk a webcam is used.
# import the necessary packages
import numpy as np
import time
import cv2
import os
import PySimpleGUI as sg
from google_speech import Speech
import system_logging as log
import numpy as np
import argparse
import pickle
import queue
import threading
import json
import time
import send as api
import dlib
import cv2
import os
# context = zmq.Context()
# footage_socket = context.socket(zmq.PUB)
# footage_socket.connect('tcp://localhost:5555')

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

result = queue.Queue()
stopped = False
step = 0
datas = queue.Queue(maxsize=0)
temp = queue.Queue(maxsize=0)
attended = {"user_id": ""}
sent = {"sent": False}
frames = queue.Queue(maxsize=10)

def adjust_gamma(image):
    frame = image
    avg = np.average(frame)
    if avg < 130:
        invGamma = 1.0 / (avg/35)
        table = np.array([((i / 255.0) ** invGamma) *
                          255 for i in np.arange(0, 256)]).astype("uint8")
        frame = cv2.LUT(frame, table)
    return frame

def sends():
    while datas.not_empty:
        if temp.not_empty:
            json = datas.get()
            icon = temp.get()
            if json["user_id"] in attended["user_id"]:
                continue
            attended.update({"user_id": json["user_id"]})
            # while not sent["sent"]:
            api.post_attendance(json, icon)


def createCLAHE(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    res = clahe.apply(frame)
    return res

def processing(image,box,id):
    try:
        clean = image.copy()
        (startX, startY, endX, endY) = box.astype("int")
        # face = cv2.resize(, (300, 300), interpolation=cv2.INTER_LANCZOS4)
        # (fH, fW) = face.shape[:2]
        # if fH <80:
        #     continue
        faceBlob = cv2.dnn.blobFromImage(
            image[startY:endY, startX:endX], 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]
        text = "ID: {}, Accu: {}%".format(name, int(proba * 100),id)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY),(endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(image, "Entity {}".format(id), (startX, y-20),cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 1)
        if (proba*100) >= 80 and name != "Guest":
            if not os.path.isdir("temp"):
                os.mkdir("temp")
            path = os.path.join("temp","{}.png".format(name))
            cv2.imwrite(path, clean[startY:endY, startX:endX])
            dict = {"user_id": name,
                    "camera_id": "5d4c276616171e2938004c72"}
            file = {"photo": open(path, 'rb')}
            temp.put(file)
            datas.put(dict)
        result.put(image)
    except Exception as e:
        print(e)
        result.put(image)

        

# @profile
def get_frame():
    try:
        vid = cv2.VideoCapture(0)
        # vid = cv2.VideoCapture('rtsp://103.247.219.34:554/user=admin&password=admin&channel=0&stream=0.sdp')
        while True:
            ret,frame = vid.read()
            if not ret:
                break
            if frames.full():
                continue
            frames.put(frame)
        vid.release()
    except Exception as e:
        pass

def detect_face(image):
    (h, w) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(image, 1.0,size=(400, 400), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    threads = []
    entity_id = 0
    for i in range(0, detections.shape[2]):
        entity_id += 1
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            threads.append(threading.Thread(target=processing(image,box,entity_id))) 
    entity_id = 0
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()



import multiprocessing

if __name__ == "__main__":
    t = threading.Thread(target=sends)
    t.daemon = True
    t.start()
    t2 = threading.Thread(target=get_frame)
    t2.daemon = True
    t2.start()
    while not stopped:
        if not frames.not_empty:
            continue
        image = frames.get()
        thread = threading.Thread(target=detect_face(image))
        thread.daemon = True
        thread.start()
        if result.not_full:
            frame = cv2.resize(result.get(),(1080,720))
            cv2.putText(frame, "Active Threads : {}".format(threading.active_count()), (0, 100),cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 1)        
            cv2.imshow("frame", frame)
        else:
            image = cv2.resize(image,(1080,720))
            cv2.putText(image, "Active Threads : {}".format(threading.active_count()), (0, 100),cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 1)
            cv2.imshow("frame",image)
            # result.clear()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stopped = True
            exit()
    cv2.destroyAllWindows()