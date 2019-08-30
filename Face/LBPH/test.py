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
fps = vid.get(cv2.CAP_PROP_FPS)
while True:
    _,frame = vid.read()
    # labels,confidences,rects = resist.predict(frame,w,h)
    # for label,confidence,rect in zip(labels,confidences,rects):
    #     print(label,confidence)
    frame = resist.adjust_brightness(frame)
    faces, boxes = resist.normalize_face_data(frame,w,h)
    # for box in boxes:
    #     resist.draw_rectangle(frame,box)
    if faces == None:
        continue
    for face,box in zip(faces,boxes):
        try:
            label, confidence = resist.predict(face)
        except Exception as e:
            print(e)
        (startX, startY, endX, endY) = box.astype("int")
        y = startY - 10 if startY - 10 > 10 else startY + 10 
        if confidence<45:
            resist.draw_rectangle(frame,box)
            resist.draw_text(frame,str(label),startX, y)
            cv2.putText(frame,"FPS : {}".format(fps),(0,100),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)        
        else:
            resist.draw_rectangle(frame,box)
            resist.draw_text(frame,str("???"),startX, y)
            cv2.putText(frame,"FPS : {}".format(fps),(0,100),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
        print(confidence)
    try:
        cv2.imshow("frame",frame)
    except Exception as e:
        print(e)
        pass
    print(label,confidence)
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break