from operator import le
import ujson as json
import glob
import os
import shutil

with open(
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/ood_3.json",
    "r",
) as f:
    ann = json.load(f)
img = ann["images"]
dstDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/images_neg"
search = f"{dstDir}/**/*"
srcDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/OOD_reject"
allInPlaceImgs = os.listdir(dstDir)
for i in img:
    filename = i["file_name"]
    if filename not in allInPlaceImgs:
        search = f"{srcDir}/**/{filename}"
        find = glob.glob(search, recursive=True)
        if len(find) == 1:
            shutil.copy(find[0], dstDir)
