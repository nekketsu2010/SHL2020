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

def load(kind, label, hold_position, sensor, axis, nfft, overlap):
    x = np.load("save/" + kind + "_" + hold_position + "_" + sensor + "_" + axis + "_" + str(nfft) + "_" + str(overlap) + ".npy")[:14813][(label == 2) | (label == 3) | (label == 5)]
    return x.reshape([-1, 1, x.shape[1], x.shape[2], 1])

val_label = np.load("../Data/センサ別npyファイル/validation/validation_Bag/validation_Bag_Label.npy")[:14813, 0].reshape([-1])

kind = "validation"

hold_position = "Bag"
Bag_user2 = np.concatenate([load(kind, val_label, hold_position, "Acc", "xy", nfft, overlap), load(kind, val_label, hold_position, "Acc", "z", nfft, overlap),\
    load(kind, val_label, hold_position, "Gyr", "xy", nfft, overlap), load(kind, val_label, hold_position, "Gyr", "z", nfft, overlap),\
        load(kind, val_label, hold_position, "Mag", "xy", nfft, overlap), load(kind, val_label, hold_position, "Mag", "z", nfft, overlap)], axis=1)

print("load complete", hold_position)

hold_position = "Hips"
Hips_user2 = np.concatenate([load(kind, val_label, hold_position, "Acc", "xy", nfft, overlap), load(kind, val_label, hold_position, "Acc", "z", nfft, overlap),\
    load(kind, val_label, hold_position, "Gyr", "xy", nfft, overlap), load(kind, val_label, hold_position, "Gyr", "z", nfft, overlap),\
        load(kind, val_label, hold_position, "Mag", "xy", nfft, overlap), load(kind, val_label, hold_position, "Mag", "z", nfft, overlap)], axis=1)

print("load complete", hold_position)

hold_position = "Torso"
Torso_user2 = np.concatenate([load(kind, val_label, hold_position, "Acc", "xy", nfft, overlap), load(kind, val_label, hold_position, "Acc", "z", nfft, overlap),\
    load(kind, val_label, hold_position, "Gyr", "xy", nfft, overlap), load(kind, val_label, hold_position, "Gyr", "z", nfft, overlap),\
        load(kind, val_label, hold_position, "Mag", "xy", nfft, overlap), load(kind, val_label, hold_position, "Mag", "z", nfft, overlap)], axis=1)

print("load complete", hold_position)

hold_position = "Hand"
Hand_user2 = np.concatenate([load(kind, val_label, hold_position, "Acc", "xy", nfft, overlap), load(kind, val_label, hold_position, "Acc", "z", nfft, overlap),\
    load(kind, val_label, hold_position, "Gyr", "xy", nfft, overlap), load(kind, val_label, hold_position, "Gyr", "z", nfft, overlap),\
        load(kind, val_label, hold_position, "Mag", "xy", nfft, overlap), load(kind, val_label, hold_position, "Mag", "z", nfft, overlap)], axis=1)

print("load complete", hold_position)

val_label = val_label[(val_label == 2) | (val_label == 3) | (val_label == 5)]
pattern_user2 = np.zeros(Bag_user2.shape[0])
for i in [2, 3, 5]:
    tmp = np.where(val_label == i)[0]
    pattern_user2[tmp[:tmp.shape[0]//10*7]] = 1

def load(kind, label, hold_position, sensor, axis, nfft, overlap):
    x = np.load("save/" + kind + "_" + hold_position + "_" + sensor + "_" + axis + "_" + str(nfft) + "_" + str(overlap) + ".npy")[14813:][(label == 2) | (label == 3) | (label == 5)]
    return x.reshape([-1, 1, x.shape[1], x.shape[2], 1])

val_label = np.load("../Data/センサ別npyファイル/validation/validation_Bag/validation_Bag_Label.npy")[14813:, 0].reshape([-1])

hold_position = "Bag"
Bag_user3 = np.concatenate([load(kind, val_label, hold_position, "Acc", "xy", nfft, overlap), load(kind, val_label, hold_position, "Acc", "z", nfft, overlap),\
    load(kind, val_label, hold_position, "Gyr", "xy", nfft, overlap), load(kind, val_label, hold_position, "Gyr", "z", nfft, overlap),\
        load(kind, val_label, hold_position, "Mag", "xy", nfft, overlap), load(kind, val_label, hold_position, "Mag", "z", nfft, overlap)], axis=1)

print("load complete", hold_position)

hold_position = "Hips"
Hips_user3 = np.concatenate([load(kind, val_label, hold_position, "Acc", "xy", nfft, overlap), load(kind, val_label, hold_position, "Acc", "z", nfft, overlap),\
    load(kind, val_label, hold_position, "Gyr", "xy", nfft, overlap), load(kind, val_label, hold_position, "Gyr", "z", nfft, overlap),\
        load(kind, val_label, hold_position, "Mag", "xy", nfft, overlap), load(kind, val_label, hold_position, "Mag", "z", nfft, overlap)], axis=1)

print("load complete", hold_position)

hold_position = "Torso"
Torso_user3 = np.concatenate([load(kind, val_label, hold_position, "Acc", "xy", nfft, overlap), load(kind, val_label, hold_position, "Acc", "z", nfft, overlap),\
    load(kind, val_label, hold_position, "Gyr", "xy", nfft, overlap), load(kind, val_label, hold_position, "Gyr", "z", nfft, overlap),\
        load(kind, val_label, hold_position, "Mag", "xy", nfft, overlap), load(kind, val_label, hold_position, "Mag", "z", nfft, overlap)], axis=1)

print("load complete", hold_position)

hold_position = "Hand"
Hand_user3 = np.concatenate([load(kind, val_label, hold_position, "Acc", "xy", nfft, overlap), load(kind, val_label, hold_position, "Acc", "z", nfft, overlap),\
    load(kind, val_label, hold_position, "Gyr", "xy", nfft, overlap), load(kind, val_label, hold_position, "Gyr", "z", nfft, overlap),\
        load(kind, val_label, hold_position, "Mag", "xy", nfft, overlap), load(kind, val_label, hold_position, "Mag", "z", nfft, overlap)], axis=1)

print("load complete", hold_position)

val_label = val_label[(val_label == 2) | (val_label == 3) | (val_label == 5)]
pattern_user3 = np.zeros(Bag_user3.shape[0])
for i in [2, 3, 5]:
    tmp = np.where(val_label == i)[0]
    pattern_user3[tmp[:tmp.shape[0]//10*7]] = 1

x_train = np.concatenate([Bag_user2[pattern_user2 == 1], Hips_user2[pattern_user2 == 1], Torso_user2[pattern_user2 == 1], Hand_user2[pattern_user2 == 1], \
    Bag_user3[pattern_user3 == 1], Hips_user3[pattern_user3 == 1], Torso_user3[pattern_user3 == 1], Hand_user3[pattern_user3 == 1]], axis=0)
x_val = np.concatenate([Bag_user2[pattern_user2 == 0], Hips_user2[pattern_user2 == 0], Torso_user2[pattern_user2 == 0], Hand_user2[pattern_user2 == 0], \
    Bag_user3[pattern_user3 == 0], Hips_user3[pattern_user3 == 0], Torso_user3[pattern_user3 == 0], Hand_user3[pattern_user3 == 0]], axis=0)

y_train = [0] * Bag_user2[pattern_user2 == 1].shape[0] + [1] * Hips_user2[pattern_user2 == 1].shape[0] + [2] * Torso_user2[pattern_user2 == 1].shape[0] + [3] * Hand_user2[pattern_user2 == 1].shape[0] + \
    [0] * Bag_user3[pattern_user3 == 1].shape[0] + [1] * Hips_user3[pattern_user3 == 1].shape[0] + [2] * Torso_user3[pattern_user3 == 1].shape[0] + [3] * Hand_user3[pattern_user3 == 1].shape[0]
y_val = [0] * Bag_user2[pattern_user2 == 0].shape[0] + [1] * Hips_user2[pattern_user2 == 0].shape[0] + [2] * Torso_user2[pattern_user2 == 0].shape[0] + [3] * Hand_user2[pattern_user2 == 0].shape[0] + \
    [0] * Bag_user3[pattern_user3 == 0].shape[0] + [1] * Hips_user3[pattern_user3 == 0].shape[0] + [2] * Torso_user3[pattern_user3 == 0].shape[0] + [3] * Hand_user3[pattern_user3 == 0].shape[0]

y_train = np.array(y_train).reshape([-1])
y_val = np.array(y_val).reshape([-1])
# x_train = np.load("保持位置分類x_train.npy")
# x_val = np.load("保持位置分類x_val.npy")
# y_train = np.load("保持位置分類y_train.npy")
# y_val = np.load("保持位置分類y_val.npy")

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
'''
def BuildModel(input_shape):
    input1 = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), strides=(1,1), padding='valid', activation='relu', kernel_initializer=he_normal())(input1)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    try:
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
    except:
        pass
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = models.Model(inputs=input1, outputs=x)    
    return x

save_folder = "/CheckPoint_" + str(nfft) + "_" + str(overlap) + "/" #保存ディレクトリを指定(後ろにスラッシュ入れてね)

def getModel():
    x1 = BuildModel(x_train[0, 0].shape)
    x2 = BuildModel(x_train[0, 1].shape)
    x3 = BuildModel(x_train[0, 2].shape)
    x4 = BuildModel(x_train[0, 3].shape)
    x5 = BuildModel(x_train[0, 4].shape)
    x6 = BuildModel(x_train[0, 5].shape)

    combined = layers.concatenate([x1.output, x2.output, x3.output, x4.output, x5.output, x6.output])

    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dense(16, activation='relu')(z)
    z = layers.Dense(4, activation='softmax')(z)

    model = models.Model(inputs=[x1.input, x2.input, x3.input, x4.input, x5.input, x6.input], outputs=z)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

# model = getModel()
model = tf.keras.models.load_model(save_folder + "model保持位置分類_" + nfft + "_" + overlap + ".hdf5")

import os
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_folder + "model保持位置分類2_" + nfft + "_" + overlap + ".hdf5", 
                                           monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
history = model.fit([x_train[:, 0], x_train[:, 1], x_train[:, 2], x_train[:, 3], x_train[:, 4], x_train[:, 5]], y_train, epochs=512, batch_size=256, \
    validation_data=([x_val[:, 0], x_val[:, 1], x_val[:, 2], x_val[:, 3], x_val[:, 4], x_val[:, 5]], y_val), callbacks=[cp_cb, es_cb])
'''
save_folder = "/CheckPoint_" + str(nfft) + "_" + str(overlap) + "/" #保存ディレクトリを指定(後ろにスラッシュ入れてね)
model = tf.keras.models.load_model(save_folder + "model保持位置分類2_" + nfft + "_" + overlap + ".hdf5")
predict = model.predict([x_val[:, 0], x_val[:, 1], x_val[:, 2], x_val[:, 3], x_val[:, 4], x_val[:, 5]])
predict_class = np.argmax(predict, axis=1)
def plot_confusion_matrix(test_y,pred_y,class_names,normalize=False, figsize=(16, 8)):
    cm = confusion_matrix(test_y,pred_y)
    # classes = class_names[unique_labels(test_y,pred_y)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label\n',
           xlabel='\nPredicted label')
    fmt = '.2f' if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,
                    i,
                    format(cm[i, j] * 100, fmt),
                    ha="center",
                    va="center",
                    color="red", fontsize=16)
    fig.tight_layout()
    return ax
plot_confusion_matrix(y_val, predict_class, ['Bag', 'Hips', 'Torso', 'Hand'], True, (8, 4))
plt.grid(False)
plt.show()
print("F-macro", f1_score(y_val, predict_class, average='macro'))
np.save("保持位置分類0623", confusion_matrix(y_val, predict_class))
