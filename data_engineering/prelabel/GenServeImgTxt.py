import os
import glob
if __name__ == "__main__":
    imgDir = "/run/user/1000/gvfs/smb-share:server=192.168.1.4,share=extract_img"
    outputFile = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/label_process/SUV.txt"
    if os.path.exists(outputFile):
        raise Exception()
    search = f"{imgDir}/**/*.JPG"
    allImg = glob.glob(search, recursive=True)
    
    allFilename = ["/".join(x.split("/")[-2:]) for x in allImg]
    allServeUrl = [f"http://0.0.0.0:8000/{x}" for x in allFilename]
    with open(outputFile, "w") as f:
        f.write("\n".join(allServeUrl))
    