# testデータをユーザごとのモデルでpredict

import numpy as np

#testデータのユーザnpyファイルを読み込み
test_user = np.load("test_user.npy").reshape([-1, 1])
print(test_user.shape)
#ユーザ2モデルでpredictした出力確率を読み込む
test2 = np.load("test_横山分類_pattern2_user2.npy")
print(test2.shape)

#ユーザ3モデルでpredictした出力確率を読み込む
test3 = np.load("test_横山分類_pattern2_user3.npy")
print(test3.shape)
index = [i for i in range(57573)]
index = np.array(index).reshape([-1, 1])
print(index.shape)
test2 = np.concatenate([index, test_user, test2], axis=1)
test3 = np.concatenate([index, test_user, test3], axis=1)

test2 = test2[test2[:, 1] == 2]
print(test2.shape)
test3 = test3[test3[:, 1] == 3]
print(test3.shape)
test = np.concatenate([test2, test3], axis=0)
test = test[np.argsort(test[:, 0])]
print(test.shape)
np.save("test_横山分類_pattern2", test[:, 2:].astype(np.float32))

print(test[:, 2:].shape)
print(test)