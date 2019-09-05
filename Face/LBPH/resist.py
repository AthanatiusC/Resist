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
import redis
import imutils

# r = redis.Redis("localhost")
labels = []
faces = []
boxes = []
predictions = []
dataset = "dataset"
pickles = os.path.join("model", "alpha.resist")
core = os.path.join("model", "core.resist")
protoPath = os.path.join("model", "deploy.prototxt")
modelPath = os.path.join("model", "face_detector.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
recognizer_model = cv2.face.LBPHFaceRecognizer_create()

log.sys("Initializing System Core")


def check_core():
    try:
        if not os.path.isfile(core):
            create_model()
        recognizer_model.read(core)
    except:
        pass

def check_cam():
    online_cams = []
    for i in range(psutil.cpu_count(logical=True)):
        vid = cv2.VideoCapture(i)
        if vid.isOpened():
            online_cams.append(i)
    return online_cams

def create_model():
    prepare_training_dataset("dataset")
    file = pickle.load(open(pickles, "rb"))
    label = file["labels"]  # DECENTRIALIZE DICTIONARY
    face = file["faces"]
    recognizer_model.train(face, np.array(label))
    recognizer_model.write(core)

def update_model():
    file = pickle.load(open(pickles, "rb"))
    label = file["labels"]  # DECENTRIALIZE DICTIONARY
    face = file["faces"]
    recognizer_model.read(core)
    recognizer_model.update(face, np.array(label))
    recognizer_model.write(core)

def draw_rectangle(img, rect):
    (startX, startY, endX, endY) = rect.astype("int")
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def createCLAHE(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    res = clahe.apply(frame)
    return res

def new_save():
    if not os.path.isfile(pickles):  # WRITE FILE IF RESIST FILE DOESN'T EXIST
        data = {"faces": faces, "labels": labels}
        file = open(pickles, "wb")
        file.write(pickle.dumps(data))
        file.close()

def save_existing():
    newlabels = labels
    newfaces = faces
    predictions.clear()
    faces.clear()
    if os.path.isfile(pickles):
        oldfaces = []
        oldlabels = []
        file = pickle.load(open(pickles, "rb"))  # LOAD EXISTING RESIST FILE
        label = file["labels"]  # DECENTRIALIZE DICTIONARY
        face = file["faces"]
        for la,fa in zip(label,face):  # APPENDING EACH LIST
            oldlabels.append(la)
            oldfaces.append(fa)
        for la,fa in zip(newlabels,newfaces):
            oldlabels.append(la)
            oldfaces.append(fa)
        data = {"faces": oldfaces, "labels": oldlabels}
        files = open(pickles, "wb")
        files.write(pickle.dumps(data))
        files.close()
        file = pickle.load(open(pickles, "rb"))

def Serialize(frame,box):
    try:
        (startX, startY, endX, endY) = box.astype("int")
        gray = createCLAHE(frame)
        equalized = cv2.resize(gray[startY:endY, startX:endX], (500, 400), interpolation=cv2.INTER_LANCZOS4)
        faces.append(equalized)
        boxes.append(box.astype("int"))
    except:
        pass

def Normalize(frame):
    try:
        h,w,c = frame.shape
        processes = []
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (
            104.0, 177.0, 123.0), swapRB=False, crop=False)  # CONVERT FRAME INTO BLOB FOR DNN INPUT
        detector.setInput(imageBlob)
        detections = detector.forward()
        for i in range(0, detections.shape[2]):  # ITERATE ALL DETECTED FACE
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                processes.append(threading.Thread(target=Serialize,args=(frame,box)))
        for process in processes:
            process.daemon=True
            process.start()
    #     return faces, boxes    ##DEFAULT EXPECTED RETURNED VALUE
    except:
        print("None")

def prepare_training_dataset(dataset):
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
                Normalize(frame)
                for face, box in zip(faces, boxes):
                    if len(faces) != 0:
                        labels.append(label)  # !!DEFAULT EXPECTED RETURN VALUE
            sys.stdout.write("\n")
            sys.stdout.flush()
            # APPEND DATASET DIR FILE NAME WITH T MEANING ITS BEEN TRAINED
            os.rename(os.path.join(dataset, dir_name),
                    os.path.join(dataset, "T"+dir_name))
    if not os.path.isfile(pickles):
        new_save()
    if os.path.isfile(pickles):
        save_existing()

def predict(face,box,frame):
    (startX, startY, endX, endY) = box.astype("int")
    label, confidence = recognizer_model.predict(face)
    y = startY - 10 if startY - 10 > 10 else startY + 10 
    draw_rectangle(frame,box)
    if confidence < 30:
        draw_text(frame,str(label)+" Accu: "+str(100-int(confidence)+15),startX, y)
    data = {"label":label,"confidence":confidence,"frame":frame}
    predictions.append(data)

##TODO : Make everything Multi threaded
log.sys("Completed!")
log.sys("Running Main System")
def main(cam):
    vid = cv2.VideoCapture(cam)
    while True:
        _,frame = vid.read()
        frame = imutils.resize(frame,width=720)
        original = frame.copy()
        start = time.time()
        avg = np.average(frame)
        if avg < 130:
            invGamma = 1.0 / (avg/35)
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            frame = cv2.LUT(frame, table)
        Normalize(frame)
        if len(faces) != 0:
            threads = []
            for face,box in zip(faces,boxes):
                t = threading.Thread(target=predict, args=(face,box,frame))
                threads.append(t)
            faces.clear()
            boxes.clear()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for prediction in predictions:
                frame = prediction["frame"]
        try:
            frm = "FPS : "+ str(int(1.0 / (time.time() - start)))
            cpu = "CPU : "+str(psutil.cpu_percent()) +"%"
            mem = "MEM : " +str(psutil.virtual_memory()[2])+"%"
            draw_text(frame,frm,0,100)
            draw_text(frame,cpu,0,120)
            draw_text(frame,mem,0,140)
            cv2.imshow("Frame",frame)
        except:
            pass
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    try:
        recognizer_model.read(core)
    except:
        create_model()
    main(0)
    # cams = check_cam()
    # p = multiprocessing.Process(target=main(0))
    # p.daemon=True
    # p.start()