import pandas as pd
import os
from zipfile import ZipFile
from tqdm import tqdm

taskFile = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/download_task.csv"
fileMeta = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/file_metadata.csv"
zipDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/temp_unzip/input"
outputDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/temp_unzip/unzip"
taskDf = pd.read_csv(taskFile)
metadataDf = pd.read_csv(fileMeta)
merged = pd.merge(taskDf, metadataDf, left_on="caseId", right_on="CaseID")
allMatchDf = merged[merged["StdDocDesc"] == merged["viewName"]]
for _, row in tqdm(allMatchDf.iterrows()):
    try:
        caseId = row["caseId"]
        zipFile = os.path.join(zipDir, str(caseId) + ".zip")
        imgFilename = row["filename"]
        vehicleDir = os.path.join(outputDir, row["vehicleType"])
        os.makedirs(vehicleDir, exist_ok=True)
        viewDir = os.path.join(vehicleDir, row["viewName"].replace(" ", ""))
        os.makedirs(viewDir, exist_ok=True)

        with ZipFile(zipFile, "r") as zipObject:
            listOfFileNames = zipObject.namelist()
            for fileName in listOfFileNames:
                if fileName == imgFilename:
                    zipObject.extract(fileName, viewDir)
                    break
    except Exception as e1:
        print(e1)
        