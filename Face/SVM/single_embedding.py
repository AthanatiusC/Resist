from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import system_logging as log
import send
import datetime
import json

dataset = "dataset"
# model = os.path.join("models","alpha.vcore")
model = os.path.join("output","embeddings.pickle")
##TODO USE CUSTOM FILES
def main(imagePath):
    start = datetime.datetime.now()
    
    log.sys("Loading face detector and recognizer")
    protoPath = os.path.join("face_detection_model", "deploy.prototxt")
    modelPath = os.path.join("face_detection_model",
                            "res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
    log.sys("Completed!")
    log.sys("Processing faces...")

    imagePaths = list(paths.list_images(os.path.join(dataset,imagePath)))
    knownEmbeddings = []
    knownNames = []
    
    total = 0

    for (i, imagePath) in enumerate(imagePaths):
        log.log("Processing image {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=1920)
        # details = json.loads(send.get_identity(name))
        # if details["status"] == "0":
        # 	continue
        # elif details["status"] == "1" and details["name"] != "":
        # 	name = details["name"]
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1
    log.sys("Serializing {} encodings...".format(total))
    end = datetime.datetime.now()
    log.sys("Training Completed!")
    log.sys("Time Consumed : {} Seconds".format(end-start))

    if os.path.isfile(model):
        file = pickle.load(open(model, "rb"))
        embdeddings = file["embeddings"]
        names = file["names"]
        file.close()

        for embedding,name in zip(embdeddings,names):
            knownEmbeddings.append(embedding)
            knownNames.append(name)

        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(model, "wb")
        f.write(pickle.dumps(data))
        f.close()
    elif not os.path.isfile(model):
        os.mkdir("models")
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(model, "wb")
        f.write(pickle.dumps(data))
        f.close()