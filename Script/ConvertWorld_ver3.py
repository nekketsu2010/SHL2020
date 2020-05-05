import math
import numpy as np
import sys
from tqdm import tqdm

def calcGloAcc(x, y, z, ori_w, ori_x, ori_y, ori_z):
    pitch = 1 - 2 * (ori_x * ori_x + ori_y * ori_y)
    yaw = 1 - 2 * (ori_y * ori_y + ori_z * ori_z)
    if pitch == 0:
        pitch = sys.float_info.epsilon
    if yaw == 0:
        yaw = sys.float_info.epsilon

    pitch = math.atan((2 * (ori_w * ori_x + ori_y * ori_z)) / pitch) #θ

    roll = 2 * (ori_w * ori_y - ori_z * ori_x)
    if roll > 1:
        roll = 1
    elif roll < -1:
        roll = -1
    roll = math.asin(roll) #φ

    yaw = math.atan((2 * (ori_w * ori_z + ori_x * ori_y)) / yaw) #ψ

    R1 = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    R2 = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R3 = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

    sensor = np.array([x, y, z]).T
    result = np.dot(R3, R2)
    result = np.dot(result, R1)
    result = np.dot(result, sensor).T
    return result.tolist()


argv = sys.argv[1]
path = '../Data/Raw/' + argv  + "/"
sensor_name = "Mag"
with open(path + sensor_name + "_x.txt") as f:
	acc_x = f.readlines()
print("ok")
with open(path + sensor_name + "_y.txt") as f:
	acc_y = f.readlines()
print("ok")
with open(path + sensor_name + "_z.txt") as f:
	acc_z = f.readlines()
print("ok")

with open(path + "Ori_w.txt") as f:
	ori_w = f.readlines()
print("ok")
with open(path + "Ori_x.txt") as f:
	ori_x = f.readlines()
print("ok")
with open(path + "Ori_y.txt") as f:
	ori_y = f.readlines()
print("ok")
with open(path + "Ori_z.txt") as f:
	ori_z = f.readlines()
print("ok")


GloAcc_x = []
GloAcc_y = []
GloAcc_z = []
for acc_x_s, acc_y_s, acc_z_s, ori_w_s, ori_x_s, ori_y_s, ori_z_s  in tqdm(zip(acc_x, acc_y, acc_z, ori_w, ori_x, ori_y, ori_z)):
    acc_x_s = acc_x_s.split(' ')
    acc_y_s = acc_y_s.split(' ')
    acc_z_s = acc_z_s.split(' ')
    ori_w_s = ori_x_s.split(' ')
    ori_x_s = ori_y_s.split(' ')
    ori_y_s = ori_z_s.split(' ')
    ori_z_s = ori_z_s.split(' ')

    j = 0
    GloAcc_x_s = ""
    GloAcc_y_s = ""
    GloAcc_z_s = ""
    for acc_x_s_s, acc_y_s_s, acc_z_s_s, ori_w_s_s, ori_x_s_s, ori_y_s_s, ori_z_s_s in zip(acc_x_s, acc_y_s, acc_z_s, ori_w_s, ori_x_s, ori_y_s, ori_z_s):
        glovalAcc = calcGloAcc(float(acc_x_s_s), float(acc_y_s_s), float(acc_z_s_s), float(ori_w_s_s), float(ori_x_s_s), float(ori_y_s_s), float(ori_z_s_s))
        if j != 0:
            GloAcc_x_s += " "
            GloAcc_y_s += " "
            GloAcc_z_s += " "
        GloAcc_x_s += str(glovalAcc[0])
        GloAcc_y_s += str(glovalAcc[1])
        GloAcc_z_s += str(glovalAcc[2])
        j += 1
    GloAcc_x.append(GloAcc_x_s)
    GloAcc_y.append(GloAcc_y_s)
    GloAcc_z.append(GloAcc_z_s)

for x in tqdm(GloAcc_x):
    with open(path + "Glo_" + sensor_name + "_x_ver3.txt", "a") as f:
        f.write(x + "\n")
print("xできた")
for y in tqdm(GloAcc_y):
    with open(path + "Glo_" + sensor_name + "_y_ver3.txt", "a") as f:
        f.write(y + "\n")
print("yできた")
for z in tqdm(GloAcc_z):
    with open(path + "Glo_" + sensor_name + "_z_ver3.txt", "a") as f:
        f.write(z + "\n")
print("zできた")
