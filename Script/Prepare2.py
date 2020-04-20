import numpy as np
import ConvertWorld
import sys

def calGlobalAcc(accels, gravities, geomagnetics):
    inR = [0] * 16
    inR = ConvertWorld.getRotationMatrix(R=inR, I=None, gravity=gravities, geomagnetic=geomagnetics)
    outR = [0] * 16
    outR = ConvertWorld.remapCoodinateSystem(inR=inR, X=1, Y=2, outR=outR)
    temp = [0] * 4
    temp[0] = accels[0]
    temp[1] = accels[1]
    temp[2] = accels[2]
    temp[3] = 0
    temp = np.reshape(temp, (4, 1))
    outR = np.reshape(outR, (4, 4))
    try:
        inv = np.linalg.inv(outR)
    except np.linalg.linalg.LinAlgError:
        inv = np.identity(4, dtype=float)
    globalValues = np.dot(inv, temp)
    return globalValues

argv = sys.argv[1]
path = '../Data/Raw/validation/' + argv + '/'
with open(path + "LAcc_x.txt") as f:
	acc_x = f.readlines()
print("ok")
with open(path + "LAcc_y.txt") as f:
	acc_y = f.readlines()
print("ok")
with open(path + "LAcc_z.txt") as f:
	acc_z = f.readlines()
print("ok")
with open(path + "Gra_x.txt") as f:
	gra_x = f.readlines()
print("ok")
with open(path + "Gra_y.txt") as f:
	gra_y = f.readlines()
print("ok")
with open(path + "Gra_z.txt") as f:
	gra_z = f.readlines()
print("ok")
with open(path + "Mag_x.txt") as f:
	mag_x = f.readlines()
print("ok")
with open(path + "Mag_y.txt") as f:
	mag_y = f.readlines()
print("ok")
with open(path + "Mag_z.txt") as f:
	mag_z = f.readlines()
print("ok")


GloAcc_x = []
GloAcc_y = []
GloAcc_z = []
i = 0
for acc_x_s, acc_y_s, acc_z_s, gra_x_s, gra_y_s, gra_z_s, mag_x_s, mag_y_s, mag_z_s in zip(acc_x, acc_y, acc_z, gra_x, gra_y, gra_z, mag_x, mag_y, mag_z):
    acc_x_s = acc_x_s.split(' ')
    acc_y_s = acc_y_s.split(' ')
    acc_z_s = acc_z_s.split(' ')
    gra_x_s = gra_x_s.split(" ")
    gra_y_s = gra_y_s.split(" ")
    gra_z_s = gra_z_s.split(" ")
    mag_x_s = mag_x_s.split(" ")
    mag_y_s = mag_y_s.split(" ")
    mag_z_s = mag_z_s.split(" ")

    j = 0
    GloAcc_x_s = ""
    GloAcc_y_s = ""
    GloAcc_z_s = ""
    for acc_x_s_s, acc_y_s_s, acc_z_s_s, gra_x_s_s, gra_y_s_s, gra_z_s_s, mag_x_s_s, mag_y_s_s, mag_z_s_s in zip(acc_x_s, acc_y_s, acc_z_s, gra_x_s, gra_y_s, gra_z_s, mag_x_s, mag_y_s, mag_z_s):
        glovalAcc = calGlobalAcc([float(acc_x_s_s), float(acc_y_s_s), float(acc_z_s_s)], [float(gra_x_s_s), float(gra_y_s_s), float(gra_z_s_s)], [float(mag_x_s_s), float(mag_y_s_s), float(mag_z_s_s)])
        if j != 0:
            GloAcc_x_s += " "
            GloAcc_y_s += " "
            GloAcc_z_s += " "
        GloAcc_x_s += str(glovalAcc[0][0])
        GloAcc_y_s += str(glovalAcc[1][0])
        GloAcc_z_s += str(glovalAcc[2][0])
        j += 1
    GloAcc_x.append(GloAcc_x_s)
    GloAcc_y.append(GloAcc_y_s)
    GloAcc_z.append(GloAcc_z_s)
    i += 1
    print(str(i) + "終わった")
for x in GloAcc_x:
    with open(path + "Glo_LAcc_x.txt", "a") as f:
        f.write(x + "\n")
print("xできた")
for y in GloAcc_y:
    with open(path + "Glo_LAcc_y.txt", "a") as f:
        f.write(y + "\n")
print("yできた")
for z in GloAcc_z:
    with open(path + "Glo_LAcc_z.txt", "a") as f:
        f.write(z + "\n")
print("zできた")
