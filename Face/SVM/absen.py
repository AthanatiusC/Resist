# YOLO object detection using a webcam
# Exact same demo as the read from disk, but instead of disk a webcam is used.
# import the necessary packages
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
import base64
import zmq
from flask_opencv_streamer.streamer import Streamer
# context = zmq.Context()
# footage_socket = context.socket(zmq.PUB)
# footage_socket.connect('tcp://localhost:5555')7

stream = Streamer(5555,False,stream_res=(1280,720))
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

def adjust_gamma(image, gamma=1.0):
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

def send_stream(image):
    retval, buffer = cv2.imencode('.png', image)
    encoded = str(base64.b64encode(buffer))
    stream.send_string(encoded)
    print("sent")

def createCLAHE(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
        res = clahe.apply(frame)
        return res

sg.ChangeLookAndFeel('Dark')
layout = 	[
		[sg.Text('Attendance GUI', size=(18,1), font=('Any',18),text_color='#1c86ee' ,justification='left')],
		# [sg.Text('NIS/NIK'), sg.In('',size=(40,1), key='NISNIK')],
        # [sg.Text('Resolution'), sg.In('1000',size=(40,1), key='Resolution'),sg.Text('px')],
        [sg.Text('Camera ID'), sg.In('0',size=(40,1), key='CAMID')],
		[sg.Text('Gamma'), sg.Slider(range=(1,5),orientation='h', resolution=.1, default_value=1, size=(15,15), key='Gamma')],
		[sg.Text('Resolution'), sg.Slider(range=(720,1920),orientation='h', resolution=.1, default_value=1000, size=(15,15), key='Resolution')],
		[sg.Text('Confidence'), sg.Slider(range=(0,1),orientation='h', resolution=.1, default_value=0.5, size=(15,15), key='Confidence')],		
        [sg.Button("Execute"), sg.Cancel()]
			]

win = sg.Window('Attendance GUI',default_element_size=(21,1),text_justification='right',auto_size_text=False).Layout(layout)
import socket
if __name__ == "__main__":
    t = threading.Thread(target=sends)
    t.daemon = True
    t.start()
    while True:
        event, values = win.Read()
        # NISNIK = values['NISNIK']
        Gamma = values['Gamma']
        CameraID = values['CAMID']
        Resolution = values["Resolution"]
        args = values

        # uid = args["NISNIK"]
        gamma = args["Gamma"]
        minconfidence = float(args["Confidence"])
        resolution = args["Resolution"]
        camid = args["CAMID"]

        if event is None or event =='Cancel':
	        exit()
        elif event == 'Execute':
            vid = cv2.VideoCapture(int(camid))
            while unlocked != True:
                _,image =vid.read()
                if not _:
                    log.err("Camera Disconnected or not detected")
                    break 
                image = imutils.resize(image, width=600)
                original = image.copy()
                image = adjust_gamma(image,float(gamma))
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
                        try:    
                            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),(0, 0, 0), swapRB=True, crop=False)
                        except:
                            continue
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
                            # log.sys(str("Verification step : {}").format(step))
                            unlocked = False
                            dict = {"user_id":name,"camera_id":"5d4c276616171e2938004c72"}
                            file = {"photo":open('temp.png', 'rb')}
                            temp.put(file)
                            datas.put(dict)
                            # step=0
                            # log.log("\n\n Found! \n Details:\n UID : {}\n Accuracy : {}%\n".format(name,int(proba*100)))
                            # hello(name)
                            # last_name = ""
                            # step+=1
                            # if step == 2:
                            #     last_name = name
                            # elif step == 3 and last_name == name:
                            #     last_name = name
                            # elif step == 4 and name == last_name:
                            #     unlocked = False
                            #     dict = {"user_id":name,"camera_id":"5d4c276616171e2938004c72"}
                            #     file = {"photo":open('temp.png', 'rb')}
                            #     temp.put(file)
                            #     datas.put(dict)
                            #     step=0
                            #     log.log("\n\n Found! \n Details:\n UID : {}\n Accuracy : {}%\n".format(name,int(proba*100)))
                            #     # hello(name)
                            #     last_name = ""
                            # else:
                            #     log.sys("Verification Failed!")
                            #     step =1
                            #     last_name = ""
                # image = imutils.resize(image, width=int(resolution))
                # stream.update(image)
                # if not stream.is_streaming():
                #     stream.start_stream()
                cv2.imshow("Image", image)

                # encoded, buffer = cv2.imencode('.jpg', image)
                # footage_socket.send(base64.b64decode(buffer))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    vid.release()
                    cv2.destroyAllWindows()                    
                    break