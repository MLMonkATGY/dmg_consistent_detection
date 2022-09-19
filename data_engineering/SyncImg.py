import shutil
import ujson as json
from collections import Counter
import os

annFile = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/label_process/SUV_ann.json"
srcDir = "/run/user/1000/gvfs/smb-share:server=192.168.1.4,share=extract_img/SUV-5Dr"
vTypeName = srcDir.split("/")[-1]
outputDir = f"/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/regularisation/{vTypeName}"
os.makedirs(outputDir, exist_ok=True)
with open(annFile, "r") as f:
    ann = json.load(f)
allClass = Counter([x["category_id"] for x in ann["annotations"]])
print(allClass)
for imgpath in ann["images"]:
    filename = "/".join(imgpath["file_name"].split("/")[-2:])
    srcPath = os.path.join(srcDir, filename)
    shutil.copy(srcPath, outputDir)
