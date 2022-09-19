import shutil
import ujson as json
from collections import Counter
import os

annFile = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/regularisation/SUV/SUV_ann.json"
srcDir = "/run/user/1000/gvfs/smb-share:server=192.168.1.4,share=extract_img/SUV-5Dr"
vTypeName = srcDir.split("/")[-1]
outputDir = f"/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/regularisation/{vTypeName}"
os.makedirs(outputDir, exist_ok=True)
with open(annFile, "r") as f:
    ann = json.load(f)
allClass = Counter([x["category_id"] for x in ann["annotations"]])
for x in ann["annotations"]:
    if x["category_id"] == 0:
        x["bbox"] = [0, 0, 1, 1]

with open("./new_SUV.json", "w") as f:
    json.dump(ann, f)
