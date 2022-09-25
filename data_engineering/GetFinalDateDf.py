import glob
import ujson as json
import pandas as pd
from tqdm import tqdm

remoteDir = (
    "/run/user/1000/gvfs/smb-share:server=192.168.1.4,share=d$/new_files_metadata"
)
outDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/finaldate"
allJsonPath = glob.glob(f"{remoteDir}/*.json", recursive=True)
allJson = []
for i in tqdm(allJsonPath):
    with open(i, "r") as f:
        raw = json.load(f)
        caseId = int(i.split("/")[-1].split(".")[0])
        for j in range(len(raw)):
            raw[j]["CaseID"] = caseId
        allJson.extend(raw)
df = pd.json_normalize(allJson)
df.to_csv(f"{outDir}/files_metadata.csv")
