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

sensor = sys.argv[1]
kind = sys.argv[2]
nfft = int(sys.argv[3])
overlap = int(sys.argv[4])
if len(sys.argv) > 6:
    hold_position = sys.argv[6]
else:
    hold_position = sys.argv[5]

def spectram(x, nfft, overlap):
    nfft = nfft
    overlap = nfft - overlap
    
    x = scipy.signal.spectrogram(x, fs=100, nfft=nfft, noverlap=overlap, nperseg=nfft)[2]
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    return x

def load_npy(sensor, kind):
    if kind == "test":
        x = np.load("../Data/センサ別npyファイル/" + kind + "/" + kind + "_" + sensor + ".npy")
    else:
        x = np.load("../Data/センサ別npyファイル/" + kind + "/" + kind + "_" + hold_position + "/" + kind + "_" + hold_position + "_" + sensor + ".npy")
    return x

def xy(x):
    x = np.sqrt(np.square(x[:, :, 0]) + np.square(x[:, :, 1]))
    return x.reshape([-1, 500])
def z(x):
    return x[:, :, 2].reshape([-1, 500])

X = load_npy(sensor, kind)
if len(sys.argv) > 6:
    X = xy(X)
else:
    X = z(X)

X = np.apply_along_axis(spectram, 1, X, nfft=nfft, overlap=overlap)

if kind == "test":
    if len(sys.argv) > 6:
        np.save("save/" + kind + "_" + sensor + "_xy_" + str(nfft) + "_" + str(overlap), X)
    else:
        np.save("save/" + kind + "_" + sensor + "_z_" + str(nfft) + "_" + str(overlap), X)
    exit()

if len(sys.argv) > 6:
    np.save("save/" + kind + "_" + hold_position + "_" + sensor + "_xy_" + str(nfft) + "_" + str(overlap), X)
else:
    np.save("save/" + kind + "_" + hold_position + "_" + sensor + "_z_" + str(nfft) + "_" + str(overlap), X)
