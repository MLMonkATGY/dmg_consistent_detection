import requests
import pandas as pd
import os
from tqdm import tqdm

taskFile = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/download_task.csv"
downloadDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data_engineering"
downloadedZip = [int(x.split(".")[0]) for x in os.listdir(downloadDir)]

df = pd.read_csv("taskFile")
for _, row in tqdm(df.iterrows()):
    caseId = row["caseId"].item()
    if caseId in downloadedZip:
        continue
    url = row["url"].item()
    r = requests.get(url, allow_redirects=True)
    with open(f"{downloadDir}/{caseId}.zip", "wb") as f:
        f.write(r.content)
