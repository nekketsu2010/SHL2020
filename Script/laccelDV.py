import numpy as np
import matplotlib.pyplot as plt
import load_npy as ln
#nums-----------------------------------------------
T = 5
Fr = 100
N_F = Fr/2
FREQ = np.linspace(0, N_F, (T*Fr)//2)
#load data---------------------------------------------------
parentpath = "../Data/センサ別npyファイル"
ld = ln.Load(parentpath)
pos = "Torso"
tlabel_tdata = ld.train_load(pos, "Label")
glacc_tdata = ld.train_load(pos, "GLAcc")
trun_index = np.where(tlabel_tdata[:,0] == 3)[0]
trun_glacc = glacc_tdata[trun_index]
twalk_index = np.where(tlabel_tdata[:,0] == 2)[0]
twalk_glacc = glacc_tdata[twalk_index]
vlabel_tdata = ld.val_load(pos, "Label")
vglacc_tdata = ld.val_load(pos, "GLAcc")
vrun_index = np.where(vlabel_tdata[:,0] == 3)[0]
vrun_glacc = glacc_tdata[vrun_index]
vwalk_index = np.where(vlabel_tdata[:,0] == 2)[0]
vwalk_glacc = glacc_tdata[vwalk_index]
#class and functions for map()---------------------------------------
class Calc():
    def calc_dv(self, num_data):
        pre_num = np.delete(num_data,-1)
        post_num = np.delete(num_data,0)
        dv = (pre_num + post_num) * (1/Fr) * (1/2)
        return dv
c = Calc()
plt.figure()
for i in range(500):
    x_rdata = np.abs(twalk_glacc[i, :, 0])
    y_rdata = np.abs(twalk_glacc[i, :, 1])
    # z_rdata = np.abs(trun_glacc[i, :, 2])
    z_rdata = x_rdata + y_rdata

    x_wdata = np.abs(vrun_glacc[i, :, 0])
    y_wdata = np.abs(vrun_glacc[i, :, 1])
    # z_wdata = np.abs(vwalk_glacc[i, :, 2])
    z_wdata = x_wdata + y_wdata
    plt.subplot(1,3,1)
    plt.ylim(-1,1)
    plt.plot(c.calc_dv(x_rdata), label = "train_Bike")
    plt.plot(c.calc_dv(x_wdata), label = "val_run")
    plt.subplot(1,3,2)
    plt.ylim(-1,1)
    plt.plot(c.calc_dv(y_rdata), label = "train_Bike")
    plt.plot(c.calc_dv(y_wdata), label = "val_run")
    plt.subplot(1,3,3)
    plt.ylim(-1,1)
    plt.plot(c.calc_dv(z_rdata), label = "train_Bike")
    plt.plot(c.calc_dv(z_wdata), label = "val_run")
    plt.draw()
    plt.legend()
    plt.pause(0.01)
    plt.clf()