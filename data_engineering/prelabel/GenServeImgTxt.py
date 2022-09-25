import os
import glob
if __name__ == "__main__":
    imgDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/OOD_reject"
    outputFile = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/label_process/ood.txt"
    # if os.path.exists(outputFile):
    #     raise Exception()
    search = f"{imgDir}/**/*.JPG"
    allImg = glob.glob(search, recursive=True)
    
    allFilename = ["/".join(x.split("/")[-3:]) for x in allImg]
    allServeUrl = [f"http://0.0.0.0:8000/{x}" for x in allFilename]
    with open(outputFile, "w") as f:
        f.write("\n".join(allServeUrl))
    