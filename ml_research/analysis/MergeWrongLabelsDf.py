import pandas as pd

allPath = [
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/wrong_label/pickup_rear/wrong_label_1.csv",
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/wrong_label/pickup_rear/wrong_label_2.csv",
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/wrong_label/pickup_rear/wrong_label_3.csv",
]
allDf = []
for dfPath in allPath:
    allDf.append(pd.read_csv(dfPath))
mergeDf = pd.concat(allDf)
mergeDf.to_csv("./wrong_label_merged.csv")
