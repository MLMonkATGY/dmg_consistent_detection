import os
import pandas as pd
import shutil
from tqdm import tqdm

srcDfPath = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/Saloon-4Dr_FrontView_cls.csv"
annDf = pd.read_csv(srcDfPath)
samplePerLabel = 3000
sampledDf = annDf.groupby(["label", "view_name"]).sample(n=samplePerLabel, replace=True)
sampledDf.drop_duplicates(inplace=True)
print(sampledDf.groupby(["view_name", "label"])["case_id"].count())
dstBaseDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vType_range"
dirName = srcDfPath.split("/")[-1].split(".")[0]
localDir = os.path.join(dstBaseDir, dirName)
if os.path.exists(localDir):
    shutil.rmtree(localDir)
os.makedirs(localDir, exist_ok=True)
for _, row in tqdm(sampledDf.iterrows()):
    dstPath = os.path.join(localDir, row["dst_filename"])
    shutil.copyfile(row["src_path"], dstPath)
