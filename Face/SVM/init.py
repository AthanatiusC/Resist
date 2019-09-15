import os

import os
import system_logging as log
import datetime

start = datetime.datetime.now()
os.system('cls' if os.name=='nt' else 'clear')
log.sys("Extracting embeddings from all dataset images...")
os.system('python embedding.py --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7')
log.sys("Training Model...")
os.system('python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle --steps=100')
end = datetime.datetime.now()
log.sys("Task Sucessfully Completed!")
log.sys("Total Time Consumed : {}".format(end-start))
log.sys("Running Face Recognition...")
os.system("python absen.py")
