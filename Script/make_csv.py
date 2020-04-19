import os
import gc
import sys

argv = sys.argv[1]
hold_position = argv

def suffix(i):
    # 6桁の０埋め
    return "{:06d}".format(i)

with open(hold_position + "/Glo_LAcc_x.txt") as f:
	glo_lacc_x = f.readlines()
print("ok")
with open(hold_position + "/Glo_LAcc_y.txt") as f:
	glo_lacc_y = f.readlines()
print("ok")
with open(hold_position + "/Glo_LAcc_z.txt") as f:
	glo_lacc_z = f.readlines()
print("ok")
with open(hold_position + "/Label.txt") as f:
	label = f.readlines()
print("ok")

#ここでLabelを一行ずつ確認し，すべて同じラベルかどうかを見る
#同じラベルでない行は使わないことにする
#ラベルごとにサンプルを管理する
NG_num = [] #使わない行を格納する配列
for i in range(0, len(label)):
	label_s = label[i].split(" ")
	for j in range(len(label_s)):
		if label_s[j].strip() != label_s[0].strip():
			print("NGIndex is " + str(i))
			NG_num.append(i)
			break

# labels = ['Stil', 'Walking', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
#フォルダの作成（なければ）
# for i in range(0, len(labels)):
# 	if not os.path.exists('/' + labels[i]):
# 		os.mkdir(labels[i])


if not os.path.isdir("Output/" + hold_position + "_LAcc"):
	os.makedirs("Output/" + hold_position + "_LAcc")
for i in range(0, len(glo_lacc_x)):
	if i in NG_num:
		continue
	label_s = label[i].split(" ")
	with open("Output/"+ hold_position + "_LAcc/Sample" + suffix(i+1) + ".csv", mode="w") as newcsv:
		newcsv.write("Label,Glo_LAcc_x,Glo_LAcc_y,Glo_LAcc_z\n")
		glo_laccx_s = glo_lacc_x[i].split(" ")
		glo_laccy_s = glo_lacc_y[i].split(" ")
		glo_laccz_s = glo_lacc_z[i].split(" ")

		for j in range(0, len(glo_laccx_s)):
			s = label_s[0].strip() + "," + glo_laccx_s[j].strip() + "," + glo_laccy_s[j].strip() + "," + glo_laccz_s[j].strip() + "\n"
			newcsv.write(s)
	print(str(i+1) + "サンプル終わった")
gc.collect()