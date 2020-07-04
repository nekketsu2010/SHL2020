import pandas as pd
import os

file_names = os.listdir("Hand/")
file_names = [tmp for tmp in file_names if ".csv" in tmp]

hold_positions = ["Bag", "Hips", "Torso", "Hand"]
for hold_position in hold_positions:
    for file_name in file_names:
        a = pd.read_csv(hold_position + "/" + file_name)
        b = pd.read_csv(hold_position + "_LAcc/" + file_name)
        b = b.drop("Label", axis=1)
        if not os.path.exists("Master_" + hold_position):
            os.makedirs("Master_" + hold_position)
        pd.concat([a, b], axis=1).to_csv("Master_" + hold_position + "/" + file_name, index=False)
        print(hold_position + "_" + file_name)