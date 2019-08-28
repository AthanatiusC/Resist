import cv2
import os
import resist
import numpy as np
import sys
# 
dataset = "dataset"
pickles = os.path.join("model","alpha.resist")
core = os.path.join("model","core.resist")
recognizer_model = cv2.face.LBPHFaceRecognizer_create()
protoPath = os.path.join("model", "deploy.prototxt")
modelPath = os.path.join("model","res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# resist.prepare_training_dataset("dataset")
# resist.create_model()

vid = cv2.VideoCapture(0)
w = vid.get(3)
h = vid.get(4)
while True:
    _,frame = vid.read()
    # labels,confidences,rects = resist.predict(frame,w,h)
    # for label,confidence,rect in zip(labels,confidences,rects):
    #     print(label,confidence)
    faces, boxes = resist.normalize_face_data(frame,w,h)
    for box in boxes:
        resist.draw_rectangle(frame,box)
    label, confidence = resist.predict(faces[0])
    print(label,confidence)
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break