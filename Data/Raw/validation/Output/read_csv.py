import numpy as np

def wrapper(args):
    return read_csv(*args)

def read_csv(hold_position, f):
    file_path = hold_position + "_LAcc/"
    #file_path = "../Data/test/"
    return np.loadtxt(file_path + f, delimiter=',', skiprows=1)