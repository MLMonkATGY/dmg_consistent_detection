from zipfile import ZipFile
import ujson as json
import os
import glob
from tqdm import tqdm
import pandas as pd

baseInputDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/temp_unzip/input"
outputDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/temp_unzip"

search = f"{baseInputDir}/**/*.zip"
extractedFiles = os.listdir(outputDir)
for zipFilePath in tqdm(glob.glob(search, recursive=True)):
    try:
        caseId = zipFilePath.split("/")[-1].split(".")[0]

        if f"{caseId}_files.json" in extractedFiles:
            continue
        with ZipFile(zipFilePath, "r") as zipObject:
            listOfFileNames = zipObject.namelist()
            for fileName in listOfFileNames:
                if fileName == "files.json":
                    # Extract a single file from zip
                    with zipObject.open(fileName, "r") as f:
                        payload = json.load(f)
                    extracFileName = os.path.join(outputDir, f"{caseId}_files.json")
                    with open(extracFileName, "w") as f2:
                        json.dump(payload, f2)
    except Exception as e1:
        print(e1)


search2 = f"{outputDir}/**/*.json"
allMetadata = []
for files in tqdm(glob.glob(search2, recursive=True)):
    try:
        caseId = int(files.split("/")[-1].split(".")[0].split("_")[0])

        with open(files, "r") as f:
            fileMetadata = json.load(f)
        for metadata in fileMetadata:
            metadata["CaseID"] = caseId
            filename = str(caseId) + "_" + str(metadata["iDOCID"]) + ".JPG"
            metadata["filename"] = filename
        allMetadata.extend(fileMetadata)
    except Exception as e1:
        print(e1)
df = pd.json_normalize(allMetadata)
df.to_csv("./file_metadata.csv")
