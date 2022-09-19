from cmath import isnan
from re import T
import pandas as pd
import os
import shutil
from tqdm import tqdm

df = pd.read_csv(
    "/home/alextay96/Desktop/workspace/dmg_price/data/complete_encoded.csv"
)
df.dropna(subset=["Sum_Insured"], inplace=True)
pgCostDf = pd.read_csv("/home/alextay96/Desktop/workspace/dmg_price/data/pg_cost.csv")
pgCostDf.dropna(inplace=True)
partGroupViewMap = {"Front_View": [10, 20, 30, 110], "Rear_View": [70, 75, 60, 80]}
baseDir = (
    "/run/user/1000/gvfs/smb-share:server=192.168.1.3,share=local_data/vm_case_img"
)

print(df.groupby("Vehicle_Type")["CaseID"].count().sort_values(ascending=False))
allTargetFilename = ["Front View.jpg", "Rear View.jpg"]
targetVehicleType = "Saloon - 4 Dr"
totalSampleLimit = 100000
df2 = set(df[df["Vehicle_Type"] == targetVehicleType]["CaseID"].tolist())
allDownloadedCase = set([int(x) for x in os.listdir(baseDir)])
intersect = list(df2.intersection(allDownloadedCase))
threshold_ratio = 0.1
allAnn = []

for srcDirName in tqdm(intersect[:totalSampleLimit]):
    fullSrcDir = os.path.join(baseDir, str(srcDirName))
    allImgView = os.listdir(fullSrcDir)
    validFilename = [x for x in allImgView if x in allTargetFilename]
    for filename in validFilename:
        viewName = filename.split(".")[0].replace(" ", "_")
        if viewName not in partGroupViewMap.keys():
            continue
        relatedPG = partGroupViewMap[viewName]
        sum = 0
        for pg in relatedPG:
            target = pgCostDf[
                (pgCostDf["CaseID"] == srcDirName) & (pgCostDf["PartGroup"] == pg)
            ]
            if target.empty:
                continue
            cost = target["Cost"].item()
            if isnan(cost):
                sum = 0
                break
            sum += cost

        caseData = df[df["CaseID"] == srcDirName]
        if caseData.empty:
            continue
        sumInsured = caseData["Sum_Insured"].item()
        if isnan(sumInsured) or isnan(sum):
            print(sumInsured)
            continue
        dmgSeverity = 0
        if sum <= sumInsured * threshold_ratio:
            dmgSeverity = 0
        else:
            dmgSeverity = 1

        dstFilename = str(srcDirName) + "_" + filename
        annImage = {
            "src_path": os.path.join(fullSrcDir, filename),
            "label": dmgSeverity,
            "case_id": srcDirName,
            "src_filename": filename,
            "dst_filename": dstFilename,
            "pg_cost": sum,
            "threshold_ratio": threshold_ratio,
            "sum_insured": sumInsured,
            "view_name": viewName,
        }
        allAnn.append(annImage)
targetDf = pd.json_normalize(allAnn)
outputDfFilename = targetVehicleType.replace(" ", "")
targetDf.to_csv(f"./{outputDfFilename}.csv")
