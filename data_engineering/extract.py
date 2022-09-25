import pandas as pd
import os
import shutil

df = pd.read_csv(
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/KFold_Saloon-4Dr_FrontView_cls_kfold_5_FrontView.csv"
)
srcDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vType_range/Saloon-4Dr_FrontView_cls"
outDir = (
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/ex2"
)
os.makedirs(outDir, exist_ok=True)
dd = (
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/temp"
)
targets = set(os.listdir(dd))
for label in [0, 1]:
    allFilename = set(df[df["label"] == label]["dst_filename"].tolist())
    allFilename = targets & allFilename
    for filname in allFilename:
        srcPath = os.path.join(srcDir, filname)
        dstPath = os.path.join(outDir, str(label))
        os.makedirs(dstPath, exist_ok=True)
        shutil.copy(srcPath, dstPath)
