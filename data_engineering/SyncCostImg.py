import shutil
import ujson as json
from collections import Counter
import os
import glob
import pandas as pd
from tqdm import tqdm

srcDir = "/run/user/1000/gvfs/smb-share:server=192.168.1.4,share=extract_img/Hatchback-5Dr/FrontView"
srcDf = pd.read_csv(
    "/home/alextay96/Desktop/workspace/dmg_price/data/complete_encoded.csv"
)
pgCostDf = pd.read_csv("/home/alextay96/Desktop/workspace/dmg_price/data/pg_cost.csv")
viewName = srcDir.split("/")[-1]
vTypeName = srcDir.split("/")[-2]

search = f"{srcDir}/**/*.JPG"
targetPG = [10, 20, 30, 110]
threshold_ratio = 0.1
allRemoteImg = glob.glob(search, recursive=True)
print(f"Current remote img files : {len(allRemoteImg)}")
casePriceRange = []
for srcFullPath in tqdm(allRemoteImg):
    caseId = int(srcFullPath.split("/")[-1].split("_")[0])
    srcFilename = srcFullPath.split("/")[-1]
    caseMetadata = srcDf[srcDf["CaseID"] == caseId]
    sumInsured = caseMetadata["Sum_Insured"].item()
    if caseMetadata.empty:
        continue
    pgCostTraceable = pgCostDf[(pgCostDf["CaseID"] == caseId)]
    if pgCostTraceable.empty:
        continue
    pgCost = pgCostDf[
        (pgCostDf["CaseID"] == caseId) & (pgCostDf["PartGroup"].isin(targetPG))
    ]["Cost"].sum()
    if pgCost > sumInsured * threshold_ratio:

        casePriceRange.append(
            {
                "src_path": srcFullPath,
                "case_id": caseId,
                "label": 1,
                "view_name": viewName,
                "pg_cost": pgCost,
                "src_filename": srcFilename,
                "dst_filename": srcFilename,
                "threshold_ratio": threshold_ratio,
                "sum_insured": sumInsured,
            }
        )
    else:
        casePriceRange.append(
            {
                "src_path": srcFullPath,
                "case_id": caseId,
                "label": 0,
                "view_name": viewName,
                "pg_cost": pgCost,
                "src_filename": srcFilename,
                "dst_filename": srcFilename,
                "threshold_ratio": threshold_ratio,
                "sum_insured": sumInsured,
            }
        )
df = pd.json_normalize(casePriceRange)
df.to_csv(f"{vTypeName}_{viewName}_cls.csv")
