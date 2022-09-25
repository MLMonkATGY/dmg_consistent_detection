import pandas as pd
import os
import shutil

srcDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vType_range/Saloon-4Dr_FrontView_cls"
allFilename = os.listdir(srcDir)
outDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/temp"

os.makedirs(outDir, exist_ok=True)
allCaseId = [int(x.split("/")[0].split("_")[0]) for x in allFilename]
df = pd.read_csv(
    "/home/alextay96/Desktop/workspace/dmg_price/data/complete_encoded.csv"
)
df = df[df["CaseID"].isin(allCaseId)]
df2 = df.groupby("Model")["CaseID"].count().sort_values(ascending=False)
df = df[df["Model"].str.contains("City")]
print(df.head(10))
allTargets = df["CaseID"].tolist()
for filename in allFilename:
    caseId = int(filename.split("/")[0].split("_")[0])
    if caseId in allTargets:
        srcPath = os.path.join(srcDir, filename)
        shutil.copy(srcPath, outDir)
