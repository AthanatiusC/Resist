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
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-ca", "--cam", required=True,help="number of cam")
ap.add_argument("-g", "--gamma", required=True,
	help="amount of gamma")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
start_time = time.time()
log.sys("Loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
log.sys("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())
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

def start(vid_num):
	log.sys("Starting video stream...")
	vs = cv2.VideoCapture(vid_num)
	unlocked = False
	step = 1
	while unlocked == False:
		ret,frame = vs.read()
		frame = imutils.resize(frame, width=1920)
		frame = adjust_gamma(frame, gamma=float(args["gamma"]))
		(h, w) = frame.shape[:2]
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)
		detector.setInput(imageBlob)
		detections = detector.forward()
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
				if fW < 20 or fH < 20:
					continue
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
				if (proba*100) >= 92 and name != "Guest":
					datas.put({"user_id":name,"camera_id":"5d4c276616171e2938004c72"})
					t = threading.Thread(target=background)
					t.daemon = True
					t.start()
		frame = imutils.resize(frame, width=1000)
		cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	cv2.destroyAllWindows()

if __name__ == "__main__":
	start(int(args["cam"]))