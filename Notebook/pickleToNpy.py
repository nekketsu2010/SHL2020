import time
import numpy as np
import pickle

start = time.time()

with open("train_Bag.binaryfile", 'rb') as f:
    np.save("train_Bag",  pickle.load(f))

print(str(time.time() - start) + "sec")
start = time.time()

with open("train_Hips.binaryfile", 'rb') as f:
    np.save("train_Hips",  pickle.load(f))

print(str(time.time() - start) + "sec")
start = time.time()

with open("train_Torso.binaryfile", 'rb') as f:
    np.save("train_Torso",  pickle.load(f))

print(str(time.time() - start) + "sec")
start = time.time()

with open("train_Hand.binaryfile", 'rb') as f:
    np.save("train_Hand",  pickle.load(f))

print(str(time.time() - start) + "sec")
start = time.time()
