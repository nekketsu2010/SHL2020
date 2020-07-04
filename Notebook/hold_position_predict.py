import numpy as np
import os
import random
import scipy
from scipy import signal
from numpy.fft import fft
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import he_normal
from tensorflow.python.keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split

import pandas as pd
import pickle

import seaborn as sns
sns.set(font='Yu Gothic')
import matplotlib.pyplot as plt

nfft = "64"
overlap = "8"
kind = "test"

test_walk_run_car_index = np.load("test_walking_run_car_index.npy")
print(test_walk_run_car_index)

def load(sensor, axis, nfft, overlap):
    if kind == "test":
        x = np.load("save/" + kind + "_" + sensor + "_" + axis + "_" + str(nfft) + "_" + str(overlap) + ".npy")
    else:
        x = np.load("save/" + kind + "_Hips_" + sensor + "_" + axis + "_" + str(nfft) + "_" + str(overlap) + ".npy")
    return x.reshape([-1, 1, x.shape[1], x.shape[2], 1])

X = np.concatenate([load("Acc", "xy", nfft, overlap), load("Acc", "z", nfft, overlap),\
    load("Gyr", "xy", nfft, overlap), load("Gyr", "z", nfft, overlap),\
        load("Mag", "xy", nfft, overlap), load("Mag", "z", nfft, overlap)], axis=1)


X = np.round(X, 5)

X = X[test_walk_run_car_index]

save_folder = "/CheckPoint_" + str(nfft) + "_" + str(overlap) + "/" #保存ディレクトリを指定(後ろにスラッシュ入れてね)

model = tf.keras.models.load_model(save_folder + "model保持位置分類2_" + nfft + "_" + overlap + ".hdf5")

predict = model.predict([X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]])

hold_positions = ["Bag", "Hips", "Torso", "Hand"]
for i in range(4):
    print(hold_positions[i], np.sum(predict[:, i] >= 0.99))