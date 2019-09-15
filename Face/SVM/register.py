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
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import numpy as np
import system_logging as log
import time
import send
import json
import threading
import queue

sg.ChangeLookAndFeel('Dark')
layout = 	[
		[sg.Text('Register Dataset', size=(18,1), font=('Any',18),text_color='#1c86ee' ,justification='left')],
		[sg.Text('NIS/NIK'), sg.In('',size=(40,1), key='NISNIK')],
        [sg.Text('Dataset Resolution'), sg.In('1000',size=(40,1), key='datasetres'),sg.Text('px')],
        [sg.Text('Resolution'), sg.In('1000',size=(40,1), key='Resolution'),sg.Text('px')],
        [sg.Text('Camera ID'), sg.In('0',size=(40,1), key='CAMID')],
		# [sg.Text('Sampling'), sg.Slider(range=(0,5),orientation='h', resolution=.1, default_value=3, size=(15,15), key='Sampling')],
		[sg.Text('Gamma'), sg.Slider(range=(1,5),orientation='h', resolution=.1, default_value=1, size=(15,15), key='Gamma')],
		[sg.Text('Resolution'), sg.Slider(range=(720,1920),orientation='h', resolution=.1, default_value=1000, size=(15,15), key='Resolution')],
		# [sg.Text('Confidence'), sg.Slider(range=(0,1),orientation='h', resolution=.1, default_value=0.5, size=(15,15), key='Confidence')],
        # [sg.Text('Output:')],
        # [sg.Output(size=(80, 10))],
        [sg.Button("Register"), sg.Cancel()]
			]
            
win = sg.Window('Register Faces',default_element_size=(21,1),text_justification='right',auto_size_text=False).Layout(layout)


os.system('cls' if os.name=='nt' else 'clear')
log.sys("Initliazing components..")


restart = False
name = ""
currentdir = os.getcwd()
datasetdir = os.path.join(currentdir,"dataset")
# detector = cv2.CascadeClassifier(os.path.join(currentdir,"haarcascade_frontalface_default.xml"))
directory = os.path.join(datasetdir,name)

protoPath = os.path.join("face_detection_model", "deploy.prototxt")
modelPath = os.path.join("face_detection_model",
    "res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
total = 0
start_time = time.time()
xs = 1 
fps = 0
max_dataset = 100
counter = 0
passes = 0
cropped = ""
# sampling = 0
X=""
Y=""
H=""
W=""


datas = queue.Queue()
temp = queue.Queue()
orig_queue = queue.Queue()
file_count_queue = queue.Queue()

time.sleep(0.1)

log.sys("Initialization completed!")
time.sleep(0.1)
log.sys("Running System...")
time.sleep(0.1)

def save(image,dirs,number,formats):
    # if int(samples) != 0:
    #     for i in range(samples):
    #         name = "{}{}.{}".format(number,i,formats)
    #         path_name = os.path.join(dirs,name)
    #         cv2.imwrite(path_name,image)
    #         log.sys("Image created : {}".format(path_name))
    # else:
    name = "{}.{}".format(number,formats)
    path_name = os.path.join(dirs,name)        
    cv2.imwrite(path_name,image)
    log.sys("Image created : {}".format(path_name))

def savelist():
    while orig_queue.not_empty:
        # saves.put(save(orig,directory,file_count,"png",sampling))
        if file_count_queue.not_empty:
            try:
                path, dirs, files = next(os.walk(directory))
            except Exception as e:
                os.mkdir(directory)
            path, dirs, files = next(os.walk(directory))
            file_count = len(files)
            if len(files) != max_dataset:
                save(orig_queue.get(),directory,file_count,"png")
            else:
                pass

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)
            
def sends():
    while datas.not_empty:
        if temp.not_empty:
            send.post_register(temp.get(),datas.get())

if __name__ == "__main__":
    t = threading.Thread(target=sends)
    t2 = threading.Thread(target=savelist)
    t.daemon = True
    t2.daemon = True
    t.start()
    t2.start()
    while True:
        event, values = win.Read()
        if event is None or event =='Cancel':
	        exit()
        elif event == 'Register':
            NISNIK = values['NISNIK']
            directory = os.path.join(datasetdir,name)
            close = False
            registered = False
            Gamma = values['Gamma']
            CameraID = values['CAMID']
            Resolution = values["Resolution"]
            # Samplings = values["Sampling"]
            DR = values["datasetres"]
            args = values
            uid = args["NISNIK"]
            name = uid

            gamma = args["Gamma"]
            # sampling = int(args["Sampling"])
            # minconfidence = float(args["Confidence"])
            resolution = args["Resolution"]
            camid = args["CAMID"]
            dr = args["datasetres"]
            if uid == "":
                log.err("NIK or NIS is required!")
            else:
                # max_dataset = 20*sampling
                currentdir = os.getcwd()
                datasetdir = os.path.join(currentdir,"dataset")
                # detector = cv2.CascadeClassifier(os.path.join(currentdir,"haarcascade_frontalface_default.xml"))
                directory = os.path.join(datasetdir,name)
                vs = VideoStream(int(camid)).start()
                try:
                    while True:
                        frame = vs.read()
                        # frame = imutils.resize(frame, width=1000)
                        frame = adjust_gamma(frame, gamma=2)
                        orig = frame.copy()
                        (h, w) = frame.shape[:2]

                        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
                        detector.setInput(imageBlob)
                        detections = detector.forward()

                        counter+=1
                        if close == True:
                                break
                        for i in range(0, detections.shape[2]):
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            confidence = detections[0, 0, i, 2]
                            if confidence > 0.7:
                                (startX, startY, endX, endY) = box.astype("int")
                                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
                                if not os.path.exists(directory):
                                    os.makedirs(directory)
                                    try:
                                        path, dirs, files = next(os.walk(directory))
                                        file_count = len(files)
                                        # saves.put(save(orig,directory,file_count,"png",sampling))
                                        orig_queue.put(orig)
                                        file_count_queue.put(file_count)

                                    except Exception as ex:
                                        log.err(ex)
                                else:
                                    path, dirs, files = next(os.walk(directory))
                                    file_count = len(files)
                                    if len(files) == max_dataset:
                                        cv2.destroyAllWindows()
                                        vs.stop()
                                        vs.stream.release()
                                        os.system('python single_embedding.py --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --uid '+name)
                                        os.system('python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle --steps=100')
                                        if registered == False:
                                            cv2.imwrite("icon.png",frame[startY+20:endY+20,startX+20:endX+20])
                                            data = {"user_id":name}
                                            im = {"photo":open("icon.png",'rb')}
                                            datas.put(data)
                                            temp.put(im)
                                        log.warn("Dataset Completed!")
                                        close = True
                                    else:
                                        try:
                                            # saves.put(save(orig,directory,file_count,"png",sampling))
                                            orig_queue.put(orig)
                                        except Exception as ex:
                                            log.err(ex)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            log.warn("Master key pressed!")
                            break
                            
                        # frame = imutils.resize(frame, width=int(resolution))
                        cv2.imshow("Frame", frame)
                    try:
                        path, dirs, files = next(os.walk(directory))
                    except:
                        os.mkdir(directory)
                    path, dirs, files = next(os.walk(directory))
                    file_count = len(files)
                    log.sys("{} face images stored".format(file_count))
                    log.sys("Cleaning up...")
                    # os.system('cls' if os.name=='nt' else 'clear')
                    vs.stop()
                    log.sys("Shutting down system...")
                    vs.stream.release()
                except Exception as e:
                    log.err(e)
                    try:
                        path, dirs, files = next(os.walk(directory))
                    except:
                        os.mkdir(directory)
                    path, dirs, files = next(os.walk(directory))                        
                    file_count = len(files)
                    if file_count < max_dataset:
                        log.sys("Only {} face images were stored".format(file_count))
                    elif file_count >= max_dataset:
                        log.sys("{} face images completely stored".format(file_count))
                    log.sys("Cleaning up...")
                    time.sleep(2)
                    # os.system('cls' if os.name=='nt' else 'clear')
                    cv2.destroyAllWindows()
                    vs.stop()
                    vs.stream.release()
                    log.sys("Shutting down system...")
                    time.sleep(0.2)
                    break