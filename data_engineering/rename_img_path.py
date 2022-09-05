import os
import shutil
import os
import ujson as json

if __name__ == "__main__":
    rawAnnFile = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/result.json"
    outputJson = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/complete.json"
    urlPrefix = r"http://0.0.0.0:8000/"
    with open(rawAnnFile, "r") as f:
        ann = json.load(f)
    annStr = json.dumps(ann, escape_forward_slashes=False)
    annStr = annStr.replace(urlPrefix, "")
    annNew = json.loads(annStr)
    with open(outputJson, "w") as f:
        json.dump(annNew, f)
