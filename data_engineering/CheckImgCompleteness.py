import pandas as pd
import glob
import os
from tqdm import tqdm


srcDir = "/home/alextay96/Desktop/workspace/extract_img"
search = "{srcDir}/**/*.JPG"
allImg = glob.glob(search, recursive=True)
allCaseId = [int(x.split("/")[-1].split("_")[0]) for x in allImg]
allDir = os.listdir(srcDir)
allVehicleType = set([x.split("_")[0] for x in allDir])
print(len(set(allCaseId)))
allValidCaseId = []
for vType in tqdm(allVehicleType):
    relatedDir = glob.glob(f"{srcDir}/{vType}/*/")
    imgsFront = []
    imgsRear = []
    if len(relatedDir) > 2:
        pass
    for dirName in tqdm(relatedDir):
        imgsPath = glob.glob(f"{dirName}/**/*.JPG", recursive=True)
        caseId = [int(x.split("/")[-1].split("_")[0]) for x in imgsPath]
        if "Front" in dirName:
            imgsFront = caseId
        else:
            imgsRear = caseId
    bothImgPresent = list(set(imgsFront) & set(imgsRear))
    allValidCaseId.extend(bothImgPresent)
print(len(allValidCaseId))
