# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import system_logging as log
import time
import os
import datetime

model = os.path.join("output","embeddings.pickle")

# def main():
start = datetime.datetime.now()
log.sys("Initializing systems..")
count = 0
total = 0

data = pickle.loads(open(model, "rb").read())
le = LabelEncoder()
labels = le.fit_transform(data["names"])
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)
f = open(os.path.join("output","recognizer.pickle"), "wb")
f.write(pickle.dumps(recognizer))
f.close()
f = open(os.path.join("output","le.pickle"), "wb")
f.write(pickle.dumps(le))
f.close()

end = datetime.datetime.now()
log.sys("Training Completed!")
log.sys("Total Time Consumed : {} Seconds".format(end-start))
