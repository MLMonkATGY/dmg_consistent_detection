import glob
import os
import pandas as pd

baseSrcDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vehicle_type"
allVType = os.listdir(baseSrcDir)
allViewName = ["Front View", "Rear View"]
group = dict()
allTask = []
for vehicleType in allVType:
    for viewName in allViewName:
        searchDir = os.path.join(baseSrcDir, vehicleType)
        search = f"{searchDir}/**/*{viewName}.jpg"
        allTargetFile = glob.glob(search, recursive=True)
        assert len(allTargetFile) > 0
        allCaseId = [int(x.split("/")[-1].split("_")[0]) for x in allTargetFile]
        groupName = f"{vehicleType}_{viewName}"
        for caseId in allCaseId:
            downloadTask = {
                "vehicleType": vehicleType,
                "viewName": viewName,
                "caseId": caseId,
                "url": f"http://10.1.1.50:4011/api/dsa/query/get_caseFiles?case_id={caseId}",
            }
            allTask.append(downloadTask)
df = pd.json_normalize(allTask)
df.to_csv("./download_task.csv")
