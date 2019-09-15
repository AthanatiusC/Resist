import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--step", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
os.system('python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle --steps={}'.format(int(args['step'])))