import os

if __name__ == "__main__":
    imgDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/prelabel"
    outputFile = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/label_process/serve_img.txt"
    allFilename = os.listdir(imgDir)
    allServeUrl = [f"http://0.0.0.0:8000/{x}" for x in allFilename]
    with open(outputFile, "w") as f:
        f.write("\n".join(allServeUrl))
    