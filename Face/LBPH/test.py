import cv2
import time
import threading
import multiprocessing
import queue
import os
import numpy as np
import pickle
import sys
from resist import Resist

# TODO: MAKE CLASS
# TODO: MAKE GUI CONTROL PANEL
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:


# class Resist(object):
#     def __init__(self):
#         self.dataset = "dataset"
#         self.pickles = os.path.join("model", "alpha.resist")
#         self.core = os.path.join("model", "core.resist")
#         self.recognizer_model = cv2.face.LBPHFaceRecognizer_create()
#         self.protoPath = os.path.join("model", "deploy.prototxt")
#         self.modelPath = os.path.join(
#             "model", "res10_300x300_ssd_iter_140000.caffemodel")
#         self.detector = cv2.dnn.readNetFromCaffe(
#             self.protoPath, self.modelPath)
#         try:
#             if not os.path.isfile(core):
#                 create_model()
#             recognizer_model.read(core)
#         except:
#             pass

#     def cut_faces(self, image, faces_coord):
#         faces = []

#         for (x, y, w, h) in faces_coord:  # Trims parts of the face
#             print("{} {} {} {}".format(x, y, w, h))
#             w_rm = int(0.2 * w / 2)
#             faces.append(image[y: y + h, x + w_rm:  x + w - w_rm])
#         return faces

#     def createCLAHE(self, frame):
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
#         res = clahe.apply(frame)
#         return res

#     def adjust_brightness(frame):
#         avg = frame.frame.mean(axis=0).mean(axis=0)[0]
#         if avg < 100:
#             frame = Resist().adjust_gamma(frame, avg/60)
#             return frame

#     def adjust_gamma(self, image, gamma):
#         invGamma = 1.0 / gamma
#         table = np.array([((i / 255.0) ** invGamma) * 255
#                           for i in np.arange(0, 256)]).astype("uint8")
#         return cv2.LUT(image, table)

#     def draw_rectangle(self, img, rect):
#         (startX, startY, endX, endY) = rect.astype("int")
#         cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

#     def draw_text(self, img, text, x, y):
#         cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.5, (0, 255, 0), 2)

#     def normalize_face_data(frame, w, h):
#         try:
#             frames = []
#             boxes = []
#             imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (
#                 104.0, 177.0, 123.0), swapRB=False, crop=False)  # CONVERT FRAME INTO BLOB FOR DNN INPUT
#             detector.setInput(imageBlob)
#             detections = detector.forward()
#             # ITERATE ALL DETECTED FACE
#             for i in range(0, detections.shape[2]):
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 confidence = detections[0, 0, i, 2]
#                 # print("{} {}".format(startY,endY),"{} {}".format(startX,endX))
#                 if confidence > 0.7:
#                     (startX, startY, endX, endY) = box.astype("int")
#                     gray = createCLAHE(frame)
#                     equalized = cv2.resize(
#                         gray[startY:endY, startX:endX], (400, 400), interpolation=cv2.INTER_CUBIC)
#                     cv2.imshow("frame 2", equalized)
#                     frames.append(equalized)
#                     # !!DEFAULT EXPECTED RETURNED VALUE
#                     boxes.append(box.astype("int"))
#             return frames, boxes
#         except:
#             return None, None

    # def prepare_training_dataset(dataset):
    #     faces = []
    #     labels = []
    #     dirs = os.listdir(dataset)
    #     for dir_name in dirs:
    #         if dir_name.startswith("T"):
    #             continue
    #         if dir_name.startswith("."):
    #             continue
    #         path, dirs, file_name = next(
    #             os.walk(os.path.join(dataset, dir_name)))
    #         total = 0
    #         label = int(dir_name)
    #         fullpath = os.path.join(dataset, dir_name)
    #         # print("Trained data : uid[{}]".format(labels[:1]))
    #         for images in file_name:
    #             frame = cv2.imread(os.path.join(fullpath, images))
    #             h, w, c = frame.shape
    #             total += 1
    #             sys.stdout.write(
    #                 "\r [ SYSTEM ] : Current id '{}' - preparing {}/{} Images".format(label, total, len(file_name)))
    #             face2, boxes = self.normalize_face_data(frame, w, h)
    #             for face, box in zip(face2, boxes):
    #                 cv2.imshow("frame",face)
    #                 cv2.waitKey(0)
    #                 if face2[0] is not None:
    #                     faces.append(face)  # !!DEFAULT EXPECTED RETURN VALUE
    #                     labels.append(label)  # !!DEFAULT EXPECTED RETURN VALUE
    #             sys.stdout.flush()
    #         # APPEND DATASET DIR FILE NAME WITH T MEANING ITS BEEN TRAINED
    #         os.rename(os.path.join(dataset, dir_name),
    #                   os.path.join(dataset, "T"+dir_name))
    #     # WRITE FILE IF RESIST FILE DOESN'T EXIST
    #     if not os.path.isfile(self.pickles):
    #         data = {"faces": faces, "labels": labels}
    #         file = open(self.pickles, "wb")
    #         file.write(pickle.dumps(data))
    #         file.close()
    #     else:  # APPEND AND WRITE FILE IF FILE EXIST

    #         # LOAD EXISTING RESIST FILE
    #         file = pickle.load(open(self.pickles, "rb"))
    #         label = file["labels"]  # DECENTRIALIZE DICTIONARY
    #         face = file["faces"]

    #         for la in label:  # APPENDING EACH LIST
    #             labels.append(la)
    #         for fa in face:
    #             faces.append(fa)

    #         data = {"faces": faces, "labels": labels}
    #         files = open(self.pickles, "wb")
    #         files.write(pickle.dumps(data))
    #         files.close()
    #         file = pickle.load(open(self.pickles, "rb"))
    #     # return faces,labels
    #     sys.stdout.write("\n [ SYSTEM ] : Completed")

    # def create_model(self):
    #     self.prepare_training_dataset("dataset")
    #     file = pickle.load(open(self.pickles, "rb"))
    #     label = file["labels"]  # DECENTRIALIZE DICTIONARY
    #     face = file["faces"]
    #     self.recognizer_model.train(face, np.array(label))
    #     self.recognizer_model.write(self.core)

    # def update_model(self):
    #     file = pickle.load(open(self.pickles, "rb"))
    #     label = file["labels"]  # DECENTRIALIZE DICTIONARY
    #     face = file["faces"]
    #     self.recognizer_model.read(self.core)
    #     self.recognizer_model.update(face, np.array(label))
    #     self.recognizer_model.write(self.core)

    # def predict(self, frame):
    #     label, confidence = self.recognizer_model.predict(frame)
    #     return label, confidence

# vid = cv2.VideoCapture(0)
# w = vid.get(3)
# h = vid.get(4)
# fps = vid.get(cv2.CAP_PROP_FPS)
# while True:
#     _,frame = vid.read()
#     frame = resist.adjust_brightness(frame)
#     faces, boxes = resist.normalize_face_data(frame,w,h)
#     if faces == None:
#         continue
#     for face,box in zip(faces,boxes):
#         try:
#             label, confidence = resist.predict(face)
#         except Exception as e:
#             print(e)
#         (startX, startY, endX, endY) = box.astype("int")
#         y = startY - 10 if startY - 10 > 10 else startY + 10 
#         if confidence<45:
#             resist.draw_rectangle(frame,box)
#             resist.draw_text(frame,str(label),startX, y)
#             cv2.putText(frame,"FPS : {}".format(fps),(0,100),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)        
#         else:
#             resist.draw_rectangle(frame,box)
#             resist.draw_text(frame,str("???"),startX, y)
#             cv2.putText(frame,"FPS : {}".format(fps),(0,100),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
#         print(confidence)
#     try:
#         cv2.imshow("frame",frame)
#     except Exception as e:
#         print(e)
#         pass
#     cv2.imshow("frame",frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         cv2.destroyAllWindows()
#         break

vid = cv2.VideoCapture(0)
Resist = Resist()
Resist.check_core()
while True:
    _,frame = vid.read()
    frame = Resist.adjust_brightness(frame)
    faces,boxes = Resist.Normalize(frame)
    if faces != None:
        for face,box in zip(faces,boxes):
            label,confidence,frame = Resist.predict(face,box,frame)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break