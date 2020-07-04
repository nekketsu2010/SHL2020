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
from tqdm import tqdm

import seaborn as sns
sns.set(font='Yu Gothic')
import matplotlib.pyplot as plt
import gc


import sys

nfft = sys.argv[1]
overlap = sys.argv[2]
kind = sys.argv[3]

file_path = "save/"

def load(sensor, axis, nfft, overlap):
    if kind == "test":
        x = np.load("save/" + kind + "_Glo_" + sensor + "_" + axis + "_" + str(nfft) + "_" + str(overlap) + ".npy")
    else:
        x = np.load("save/" + kind + "_Hips_Glo_" + sensor + "_" + axis + "_" + str(nfft) + "_" + str(overlap) + ".npy")
    return x.reshape([-1, 1, x.shape[1], x.shape[2], 1])

X = np.concatenate([load("Acc", "xy", nfft, overlap), load("Acc", "z", nfft, overlap),\
    load("Gyr", "xy", nfft, overlap), load("Gyr", "z", nfft, overlap),\
        load("Mag", "xy", nfft, overlap), load("Mag", "z", nfft, overlap)], axis=1)


# round5する
X = np.round(X, 5)

save_folder = "/CheckPoint_" + str(nfft) + "_" + str(overlap) + "/" #保存ディレクトリを指定(後ろにスラッシュ入れてね)
if len(sys.argv) > 4:
    model = tf.keras.models.load_model(save_folder + "model_user" + sys.argv[4] + "_" + nfft + "_" + overlap + "_2.hdf5")
else:
    model = tf.keras.models.load_model(save_folder + "model_" + nfft + "_" + overlap + "_2.hdf5.old")
predict = model.predict([X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]])

if len(sys.argv) > 4:
    if kind == "test":
        np.save("../" + kind + "_横山分類_pattern2_user" + sys.argv[4], predict)
    else:
        np.save("../" + kind + "_Hips_横山分類_pattern2user" + sys.argv[4], predict)
else:
    if kind == "test":
        np.save("../" + kind + "_乗り物5分類", predict)
    else:
        np.save("../" + kind + "_Hips_乗り物5分類", predict)

# predict = np.argmax(predict, axis=1).reshape([-1, 1])
# np.save(kind + "_5分類", predict)

# for i in range(1, 4):
#     print("user" + str(i), np.sum(predict == i))