import numpy as np

user2 = np.load("validation_Hips_横山分類_pattern2_user2.npy")
user3 = np.load("validation_Hips_横山分類_pattern2_user3.npy")

val = np.concatenate([user2[:14813], user3[14813:]], axis=0)
np.save("validation_Hips_横山分類_pattern2", val)