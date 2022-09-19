from copy import copy
import shutil
import pandas as pd
import os
from PIL import Image
import glob
from ml_research.params.PriceRangeParams import PriceRangeParams

import shutil

srcDfPath = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/KFold_Saloon-4Dr_kfold_5_Front_View.csv"
imgBaseDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vehicle_type/Saloon-4Dr"
outputDir = (
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/temp"
)
search = f"{PriceRangeParams.rejLabelDir}/**/*.csv"
allDfPath = glob.glob(search, recursive=True)
allDf = []
for dfPath in allDfPath:
    allDf.append(pd.read_csv(dfPath))
if len(allDf) > 0:

    rejDf = pd.concat(allDf)
    rejFilename = rejDf["dst_filename"].tolist()
srcDf = pd.read_csv(srcDfPath)
# srcDf = srcDf[srcDf["label"] == 1]
srcDf = srcDf[srcDf["kfold"] == 1]
print(len(srcDf))
# srcDf = srcDf[~srcDf["dst_filename"].isin(rejFilename)]
# print(f"Rejected {len(rejFilename)} images")
print(len(srcDf))
trainNum = len(srcDf[srcDf["train_test"] == "train"])
testNum = len(srcDf[srcDf["train_test"] == "test"])

print(f"Train row : {trainNum}")
print(f"Test row : {testNum}")

if os.path.exists(outputDir):
    shutil.rmtree(outputDir)
os.makedirs(outputDir, exist_ok=True)
# wrongLabelDf = pd.read_csv(wrongLabelPath)["dst_filename"].tolist()


print(len(srcDf))
# srcDf = srcDf[~srcDf["dst_filename"].isin(wrongLabelDf)]

for _, row in srcDf.iterrows():
    srcPath = os.path.join(imgBaseDir, row["dst_filename"])
    shutil.copy(srcPath, outputDir)
