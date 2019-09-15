import pickle

e = open("output/embeddings.pickle","rb")
y = pickle.load(e)
print(y)
