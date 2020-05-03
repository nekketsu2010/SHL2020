import os
import gc
import sys

argv = sys.argv[1]
hold_position = argv

def suffix(i):
    # 6桁の０埋め
    return "{:06d}".format(i)

with open("Glo_LAcc_x.txt") as f:
	glo_lacc_x = f.readlines()
print("ok")
with open("Glo_LAcc_y.txt") as f:
	glo_lacc_y = f.readlines()
print("ok")
with open("Glo_LAcc_z.txt") as f:
	glo_lacc_z = f.readlines()
print("ok")


if not os.path.isdir("Output"):
	os.makedirs("Output")
for i in range(0, len(glo_lacc_x)):
	with open("Output/Sample" + suffix(i+1) + ".csv", mode="w") as newcsv:
		newcsv.write("Glo_LAcc_x,Glo_LAcc_y,Glo_LAcc_z\n")
		glo_laccx_s = glo_lacc_x[i].split(" ")
		glo_laccy_s = glo_lacc_y[i].split(" ")
		glo_laccz_s = glo_lacc_z[i].split(" ")

		for j in range(0, len(glo_laccx_s)):
			s = glo_laccx_s[j].strip() + "," + glo_laccy_s[j].strip() + "," + glo_laccz_s[j].strip() + "\n"
			newcsv.write(s)
	print(str(i+1) + "サンプル終わった")
gc.collect()