# USAGE
# python recognize.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle --image images/adrian.jpg

# import the necessary packages
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

vid = cv2.VideoCapture(0)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-g", "--gamma", type=float,default=1.5,
	help="set gamma intensity")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# load our serialized face detector from disk
log.sys("Loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# detector_face = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# fa = FaceAligner(predictor, desiredFaceWidth=128)


# load our serialized face embedding model from disk
log.sys("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
unlocked = False
thumbnail_created = False

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())
passes = 0
step = 0
datas = queue.Queue(maxsize=0)
temp = queue.Queue(maxsize=0)

def convert_json(uid,cam_id,photo):
	dicts = json.dumps(dict)
	return dicts

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)


def hello(name):
	sox_effects = ("speed", "1.1")
	speech = Speech("Hey there! {}!, Welcome to 10th Vocational Highschools!".format(name), "en")
	speech.play(sox_effects)


def sends():
    while datas.not_empty:
        if temp.not_empty:
            send.post_attendance(datas.get(),temp.get())

if __name__ == "__main__":
    t = threading.Thread(target=sends)
    t.daemon = True
    t.start()
    while unlocked != True:
        _,image =vid.read()
        if not _:
            log.err("Camera Disconnected")
            break 
        image = imutils.resize(image, width=1920)
        original = image.copy()
        image = adjust_gamma(image,float(args["gamma"]))
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
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
                cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                # cv2.putText(original, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                if (proba*100) >= 93 and name != "Guest":
                    log.sys(str("Verification step : {}").format(step))		
                    step+=1
                    if step == 2:
                        last_name = name
                    elif step == 3 and last_name == name:
                        last_name = name
                    elif step == 4 and name == last_name:
                        unlocked = False
                        dict = {"user_id":name,"camera_id":"5d4c276616171e2938004c72"}
                        file = {"photo":open('temp.png', 'rb')}
                        temp.put(file)
                        datas.put(dict)
                        step=0
                        log.log("\n\n Found! \n Details:\n UID : {}\n Accuracy : {}%\n".format(name,int(proba*100)))
                        # hello(name)
                        last_name = ""
                    else:
                        log.sys("Verification Failed!")
                        step =1
                        last_name = ""
        image = imutils.resize(image, width=1000)
        cv2.imshow("Image", image)
        # original = imutils.resize(original, width=1000)
        # cv2.imshow("Image", original)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break