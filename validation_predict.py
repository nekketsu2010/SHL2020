import numpy as np

val2 = np.load("validation_Hips_横山分類_pattern2_user2.npy")[:14813]
val3 = np.load("validation_Hips_横山分類_pattern2_user3.npy")[14813:]

np.save("validation_Hips_横山分類_pattern2", np.concatenate([val2, val3], axis=0))
