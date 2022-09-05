import shutil
import os
import ujson as json
import glob
from tqdm import tqdm

if __name__ == "__main__":
    rawAnnFile = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/complete.json"
    srcImgDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/prelabel"
    dstDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/images"
    remoteFileDir = ""
    with open(rawAnnFile, "r") as f:
        ann = json.load(f)
    for imgInfo in tqdm(ann["images"]):
        filename = imgInfo["file_name"]
        srcFile = f"{srcImgDir}/{filename}"
        dstFilename = os.path.join(dstDir, filename)
        shutil.copyfile(srcFile, dstFilename)
