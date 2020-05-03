import numpy as np
import sys
from tqdm import tqdm

def calcGloAcc(x, y, z, ori_w, ori_x, ori_y, ori_z):
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, 0] = 1 - 2 * (np.square(ori_y) + np.square(ori_z))
    rotation_matrix[0, 1] = 2 * (ori_x * ori_y - ori_w * ori_z)
    rotation_matrix[0, 2] = 2 * (ori_x * ori_z + ori_w * ori_y)
    rotation_matrix[1, 0] = 2 * (ori_x * ori_y + ori_w * ori_z)
    rotation_matrix[1, 1] = 1 - 2 * (np.square(ori_x) + np.square(ori_z))
    rotation_matrix[1, 2] = 2 * (ori_y * ori_z - ori_w * ori_x)
    rotation_matrix[2, 0] = 2 * (ori_x * ori_z - ori_w * ori_y)
    rotation_matrix[2, 1] = 2 * (ori_y * ori_z + ori_w * ori_x)
    rotation_matrix[2, 2] = 1 - 2 * (np.square(ori_x) + np.square(ori_y))

    sensor = np.array([x, y, z]).T
    result = np.dot(rotation_matrix, sensor).T
    return result.tolist()


argv = sys.argv[1]
path = '../Data/Raw/' + argv  + "/"
with open(path + "LAcc_x.txt") as f:
	acc_x = f.readlines()
print("ok")
with open(path + "LAcc_y.txt") as f:
	acc_y = f.readlines()
print("ok")
with open(path + "LAcc_z.txt") as f:
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
    with open(path + "Glo_LAcc_x_ver2.txt", "a") as f:
        f.write(x + "\n")
print("xできた")
for y in tqdm(GloAcc_y):
    with open(path + "Glo_LAcc_y_ver2.txt", "a") as f:
        f.write(y + "\n")
print("yできた")
for z in tqdm(GloAcc_z):
    with open(path + "Glo_LAcc_z_ver2.txt", "a") as f:
        f.write(z + "\n")
print("zできた")
