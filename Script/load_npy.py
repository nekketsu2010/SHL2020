# keyを引数に入れることでポジション毎センサ毎のデータを取り出せます
# sensor_keys がセンサを指定するときに使うkeyです
# position_keys = がポジションを指定するときに使うkeyです
# sensor_keys= ['Acc', 'GAcc', 'GLAcc', 'Gra', 'Gyr', 'Label', 'LAcc', 'Mag', 'Ori', 'Prs']
# position_keys = ['Bag', 'Hand', 'Hips', 'Torso']
# ファイル構造がこうなってること想定して作りました
# data-
#     |-test
#     |-train
#         |-Bag
#         |-Hand
#         |-Hips
#         |-Torso
#     |-validation
#         |-Bag
#         |-Hand
#         |-Hips
#         |-Torso
# こうなっているときの、dataまでのファイルパスをparentpathとして
# loader = Load(parentpath)
# としてもらうと読み込めます
# train_dataの取得,validation_dataの取得は以下のようにして出来ます
# train,validationは,第1引数がポジション,第2引数がセンサ,第3引数を(returnlabel = True)にしてもらうと、
# Nanがあった場合にどのインデックスが残ってるかを取得できます.デフォルトでreturnlabel=Falseになっているので、
# Trueを指定しなければ返ってくるのはsensorのデータだけです
# data = loader.val_load('Bag','Acc')
# data, index = loader.train_load('Bag','Acc', returnlabel = True)
# testは,第1引数センサ,第2引数を(returnlabel = True)にしてもらうと、
# Nanがあった場合にどのインデックスが残ってるかを取得できます.デフォルトでreturnlabel=Falseになっているので、
# Trueを指定しなければ返ってくるのはsensorのデータだけです
# data = loader.test_load('Acc')


import numpy as np
class Load():
    def __init__(self,parentpath):
        self.parentpath = parentpath

    def count_reset(self):
        self.count_num = 0
        self.index_list = []

    def train_load(self, poskey, senskey, returnlabel = False):
        
        position = {'Bag':'/train_Bag', 'Hand':'/train_Hand',\
             'Hips':'/train_Hips', 'Torso':'/train_Torso'}
        sensors = {'Acc':'_Acc.npy', 'GAcc':'_Glo_Acc.npy', 'GLAcc':'_Glo_Lacc.npy', 'Gra':'_Gra.npy', 'Gyr':'_Gry.npy',\
             'Label':'_Label.npy', 'LAcc':'_LAcc.npy', 'Mag':'_Mag.npy', 'Ori':'_Ori.npy', 'Prs':'_Pressure.npy'}
        parentpath = self.parentpath
        fulpath = self.parentpath + "/train" + position[poskey] + position[poskey] + sensors[senskey]
        data = np.load(fulpath)
        self.count_reset()
        data = np.array(list(map(self.ditect_nan,data)))
        if returnlabel == False:
            return data
        else:
            return data, np.array(self.index_list)

    
    def val_load(self, poskey, senskey, returnlabel = False):
        position = {'Bag':'/validation_Bag', 'Hand':'/validation_Hand',\
             'Hips':'/validation_Hips', 'Torso':'/validation_Torso'}
        sensors = {'Acc':'_Acc.npy', 'GAcc':'_Glo_Acc.npy', 'GLAcc':'_Glo_Lacc.npy', 'Gra':'_Gra.npy', 'Gyr':'_Gyr.npy',\
             'Label':'_Label.npy', 'LAcc':'_LAcc.npy', 'Mag':'_Mag.npy', 'Ori':'_Ori.npy', 'Prs':'_Pressure.npy'}
        parentpath = self.parentpath
        fulpath = self.parentpath + "/validation" + position[poskey] + position[poskey] + sensors[senskey]
        data = np.load(fulpath)
        self.count_reset()
        data = np.array(list(map(self.ditect_nan,data)))
        if returnlabel == False:
            return data
        else:
            return data, np.array(self.index_list)

    def test_load(self, senskey, returnlabel = False):
        sensors = {'Acc':'_Acc.npy', 'GAcc':'_Glo_Acc.npy', 'GLAcc':'_Glo_Lacc.npy', 'Gra':'_Gra.npy', 'Gyr':'_Gyr.npy',\
             'Label':'_Label.npy', 'LAcc':'_LAcc.npy', 'Mag':'_Mag.npy', 'Ori':'_Ori.npy', 'Prs':'_Pressure.npy'}
        parentpath = self.parentpath
        fulpath = self.parentpath + "/test/test" + sensors[senskey]
        data = np.load(fulpath)
        self.count_reset()
        data = np.array(list(map(self.ditect_nan,data)))
        if returnlabel == False:
            return data
        else:
            return data, np.array(self.index_list)

    def ditect_nan(self, data):
        self.count_num +=1
        if np.sum(np.isnan(data)) == 0:
            self.index_list.append(self.count_num)
            return data

    

