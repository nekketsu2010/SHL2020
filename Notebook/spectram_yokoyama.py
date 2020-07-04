#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


file_path = "../Data/センサ別npyファイル"

def norm(x):
    x = np.sqrt(np.square(x[:, :, 0]) + np.square(x[:, :, 1]) + np.square(x[:, :, 2]))
    return x.reshape([-1, 500])
def spectrogtam(x):
    nfft = 256
    overlap = nfft - 6
    
    x = scipy.signal.spectrogram(x, fs=100, nfft=nfft, noverlap=overlap, nperseg=nfft)[2]
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    return x
def load_npy(sensor=sensor, kind="train"):
    x = np.load(file_path + "/" + kind + "/" + kind + "_Hips/" + kind + "_Hips_Glo_" + sensor + "_ver3.npy")
    x = norm(x)
    x = np.apply_along_axis(spectrogtam, 1, x)
    return x


# In[3]:


def load(sensor):
    return np.load("save/train_Hips_" + sensor + "_256.npy")

X_train = load(sensor)


# In[4]:


pattern_file = np.load("validation_pattern2.npy").reshape([-1]) #パターンファイルのパス入れてくれ
pattern_file.shape


# In[5]:


val_label = np.load(file_path + "/validation/validation_Hips/validation_Hips_Label.npy")[:, 0].reshape([-1])
val_label -= 1
val_label.shape


# In[6]:


def load(sensor):
    return load_npy(sensor, "validation")

X_val = load(sensor)
X_val = X_val.reshape([-1, 1, X_val.shape[1], X_val.shape[2], 1])

X_val.shape


# In[9]:


X_train = np.round(X_train, 5)
X_val = np.round(X_val, 5)

X_train.shape, X_val.shape


# In[10]:


X_val_pattern0 = X_val[pattern_file == 0]
X_val_pattern1 = X_val[pattern_file == 1]
X_val_pattern2 = X_val[pattern_file == 2]

X_val_pattern0.shape, X_val_pattern1.shape, X_val_pattern2.shape


# In[11]:


def separateData(x, y):
    pattern = np.zeros(x.shape[0])
    for i in range(8):
        tmp = np.where(y == i)[0]
        pattern[tmp[tmp.shape[0]//10*7:]] = 1
        print(tmp)
        
    x_train = x[pattern == 0]
    x_test = x[pattern == 1]
    y_train = y[pattern == 0]
    y_test = y[pattern == 1]
    return x_train, x_test, y_train, y_test


# In[13]:


train_label = np.load(file_path + "/train/train_Hips/train_Hips_Label.npy")[:, 0].reshape([-1])
X_train = np.delete(X_train, 120845, axis=0)
train_label = np.delete(train_label, 120845, axis=0)
train_label -= 1
X_train.shape


# In[14]:


# x_train, x_val, y_train, y_val = separateData(X_train, train_label)
# x_train.shape, x_val.shape


# In[20]:


def BuildModel(input_shape):
    input1 = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), strides=(1,1), padding='valid', activation='relu', kernel_initializer=he_normal())(input1)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(8, activation='softmax')(x)
    x = models.Model(inputs=input1, outputs=x)    
    return x


# In[17]:




# In[21]:

# model = tf.keras.models.load_model("/" + sensor + "/model1.hdf5")
model = BuildModel(X_train[0, 0].shape)

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


save_folder = "/" + sensor + "/" #保存ディレクトリを指定(後ろにスラッシュ入れてね)

import os
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_folder + "model1.hdf5", 
                                           monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
history = model.fit(X_train[:, 0], train_label, epochs=256, batch_size=256, validation_data=(X_val_pattern1[:, 0], val_label[pattern_file == 1]), callbacks=[cp_cb, es_cb])


# In[ ]:


# # Plot training & validation accuracy values
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# # plt.ylim((0, 3.5))
# plt.show()


# In[ ]:




