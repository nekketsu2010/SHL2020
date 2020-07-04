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
kind = "validation"

def load(sensor, axis, nfft, overlap):
    if kind == "test":
        x = np.load("save/" + kind + "_Glo_" + sensor + "_" + axis + "_" + str(nfft) + "_" + str(overlap) + ".npy")
    else:
        x = np.load("save/" + kind + "_Hips_Glo_" + sensor + "_" + axis + "_" + str(nfft) + "_" + str(overlap) + ".npy")
    return x.reshape([-1, 1, x.shape[1], x.shape[2], 1])

X = np.concatenate([load("Acc", "xy", nfft, overlap), load("Acc", "z", nfft, overlap),\
    load("Gyr", "xy", nfft, overlap), load("Gyr", "z", nfft, overlap),\
        load("Mag", "xy", nfft, overlap), load("Mag", "z", nfft, overlap)], axis=1)

# # 10Hzまで使う
# X = X[:, :, :7]

# round5する
X = np.round(X, 5)

# データをユーザごとに分ける
X_user2 = X[:14813]
X_user3 = X[14813:]
del X

# trainデータも読み込む
print("trainデータ読み込み")
kind = "train"
X = np.concatenate([load("Acc", "xy", nfft, overlap), load("Acc", "z", nfft, overlap),\
    load("Gyr", "xy", nfft, overlap), load("Gyr", "z", nfft, overlap),\
        load("Mag", "xy", nfft, overlap), load("Mag", "z", nfft, overlap)], axis=1)

user1_label = np.load("../Data/センサ別npyファイル/train/train_Hips/train_Hips_Label.npy")[:, 0].reshape([-1])
pattern_user1 = np.zeros(X.shape[0])
for i in range(8):
    tmp = np.where(user1_label == (i+1))[0]
    tmp = np.random.choice(tmp, tmp.shape[0] // 10, replace=False)
    pattern_user1[tmp] = 1

X = X[pattern_user1 == 1]
X = np.round(X, 5)

# 移動状態のラベルを読み込む
user2_label = np.load("../Data/センサ別npyファイル/validation/validation_Hips/validation_Hips_Label.npy")[:14813, 0].reshape([-1])
user3_label = np.load("../Data/センサ別npyファイル/validation/validation_Hips/validation_Hips_Label.npy")[14813:, 0].reshape([-1])

# 移動状態ラベルがバランス良く訓練データ・検証データに入り込むようにパターン行列の作成
pattern_user2 = np.zeros(14813)
for i in range(8):
    tmp = np.where(user2_label == (i+1))[0]
    pattern_user2[tmp[:tmp.shape[0]//10*7]] = 1
    print(tmp[:tmp.shape[0]//10*7].shape)

# 移動状態ラベルがバランス良く訓練データ・検証データに入り込むようにパターン行列の作成
pattern_user3 = np.zeros(13872)
for i in range(8):
    tmp = np.where(user3_label == (i+1))[0]
    pattern_user3[tmp[:tmp.shape[0]//10*7]] = 1
    print(tmp[:tmp.shape[0]//10*7].shape)

np.save("ユーザ3分類ラベル", np.concatenate([user2_label[pattern_user2 == 0], user3_label[pattern_user3 == 0]], axis=0))

# ラベル作成
y_train = [0] * X.shape[0] + [1] * X_user2[pattern_user2 == 1].shape[0] + [2] * X_user3[pattern_user3 == 1].shape[0]
y_train = np.array(y_train).reshape([-1])
y_val = [1] * X_user2[pattern_user2 == 0].shape[0] + [2] * X_user3[pattern_user3 == 0].shape[0]
y_val = np.array(y_val).reshape([-1])

print(y_train.shape, y_val.shape)

# 学習データ作成
x_train = np.concatenate([X, X_user2[pattern_user2 == 1], X_user3[pattern_user3 == 1]], axis=0)
x_val = np.concatenate([X_user2[pattern_user2 == 0], X_user3[pattern_user3 == 0]], axis=0)
del X_user2, X_user3, X


def BuildModel(input_shape):
    input1 = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (2, 5), strides=(1,1), padding='valid', activation='relu', kernel_initializer=he_normal())(input1)
    x = layers.Conv2D(32, (2, 5), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((1, 3))(x)
    x = layers.BatchNormalization()(x)
    try:
        x = layers.Conv2D(64, (2, 5), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (2, 5), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((1, 3))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, (2, 5), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (2, 5), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((1, 3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (2, 5), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (2, 5), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((1, 3))(x)
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
    z = layers.Dense(3, activation='softmax')(z)

    model = models.Model(inputs=[x1.input, x2.input, x3.input, x4.input, x5.input, x6.input], outputs=z)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

model = getModel()

import os
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_folder + "modelユーザ3分類0620_" + nfft + "_" + overlap + ".hdf5", 
                                           monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
history = model.fit([x_train[:, 0], x_train[:, 1], x_train[:, 2], x_train[:, 3], x_train[:, 4], x_train[:, 5]], y_train, epochs=512, batch_size=256, \
    validation_data=([x_val[:, 0], x_val[:, 1], x_val[:, 2], x_val[:, 3], x_val[:, 4], x_val[:, 5]], y_val), callbacks=[cp_cb, es_cb])


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

model = tf.keras.models.load_model(save_folder + "modelユーザ3分類0620_" + nfft + "_" + overlap + ".hdf5")
predict = model.predict([x_val[:, 0], x_val[:, 1], x_val[:, 2], x_val[:, 3], x_val[:, 4], x_val[:, 5]])
np.save("ユーザ3分類出力確率", predict)
np.save("ユーザ3分類答え", y_val)
predict = np.argmax(predict, axis=1)

plot_confusion_matrix(y_val, predict, ['user1', 'user2', 'user3'], True, (8, 4))
plt.grid(False)
plt.show()
print("F-macro", f1_score(y_val, predict, average='macro'))
np.save("user_classify_confusion_matrix_0619", confusion_matrix(y_val, predict))