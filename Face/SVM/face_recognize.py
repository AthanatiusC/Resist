# USAGE
# python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from google_speech import Speech
import multiprocessing as mp
from playsound import playsound
from imutils.video import FPS
import system_logging as log
import numpy as np
import threading
import argparse
import imutils
import pickle
import queue
import send
import time
import cv2
import os

os.system('cls' if os.name=='nt' else 'clear')

start_time = time.time()
# log.sys("Loading face detector...")
protoPath = os.path.join("face_detection_model", "deploy.prototxt")
modelPath = os.path.join("face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# log.sys("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nnq4.small2.v1.t7")
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())
tracked = queue.Queue(maxsize=0)
datas=queue.Queue(maxsize=0)

def background():
	while datas.not_empty:
		tracked.put(send.track(datas.get()))
		tracked.task_done()
		time.sleep(3)

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def hello(name):
	sox_effects = ("speed", "1.1")
	speech = Speech("Hey there! {}!, Welcome to 10th Vocational Highschools!".format(name), "en")
	speech.play(sox_effects)

data = []

def predict(face):
	# cv2.imshow("Frame",face)
	# cv2.waitKey(0)
	faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
	embedder.setInput(faceBlob)
	vec = embedder.forward()
	preds = recognizer.predict_proba(vec)[0]
	j = np.argmax(preds)
	proba = preds[j]
	name = le.classes_[j]
	text = "{}: {:.2f}%".format(name, proba * 100)
	data.append(text)
	if (proba*100) >= 92 and name != "Guest":
		datas.put({"user_id":name,"camera_id":"5d4c276616171e2938004c72"})

def detect_face(frame):
	(h, w) = frame.shape[:2]
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()
	faces = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.7:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			if fW < 20 or fH < 20:
				continue
			faces.append(face)
	with mp.Pool(mp.cpu_count()) as p:
		faces = p.map(predict,(faces))

def start(vid_num):
	log.sys("Starting video stream...")
	vs = cv2.VideoCapture(vid_num)
	unlocked = False
	step = 1
	while unlocked == False:
		start = time.time()
		ret,frame = vs.read()
		# frame = imutils.resize(frame, width=1920)
		frame = adjust_gamma(frame, 2)
		detect_face(frame)
		for text in data:
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)	
		# frame = imutils.resize(frame, width=1000)
		# print(int(1.0 / (time.time() - start)))
		cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	cv2.destroyAllWindows()

if __name__ == "__main__":
	t = threading.Thread(target=background)
	t.daemon = True
	t.start()
	start(int(0))