# testデータをユーザごとのモデルでpredict

import numpy as np

#testデータのユーザnpyファイルを読み込み
test_user = np.load("test_user.npy").reshape([-1, 1])

#ユーザ2モデルでpredictした出力確率を読み込む
test2 = np.load("test_安部分類_pattern2_user2.npy")

#ユーザ3モデルでpredictした出力確率を読み込む
test3 = np.load("test_安部分類_pattern2_user3.npy")

index = [i for i in range(57573)]
index = np.array(index).reshape([-1, 1])

test2 = np.concatenate([index, test_user, test2], axis=1)
test3 = np.concatenate([index, test_user, test3], axis=1)

test2 = test2[test2[:, 1] == 2]
test3 = test3[test3[:, 1] == 3]

test = np.concatenate([test2, test3], axis=0)
test = test[np.argsort(test[:, 0])]
np.save("test_安部分類_pattern2", test[:, 2:].astype(np.float32))
