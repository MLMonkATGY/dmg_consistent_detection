import glob
import shutil
import os
from tqdm import tqdm

if __name__ == "__main__":
    srcDir = "/home/alextay96/Desktop/workspace/scrape_mrm/vm_case_img"
    outputDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/prelabel"
    search = f"{srcDir}/**/*.jpg"
    allImgs = glob.glob(search, recursive=True)
    for srcImgPath in tqdm(allImgs):
        elems = "_".join(srcImgPath.split("/")[-2:])
        newFilename = elems.replace(" ", "")
        dstFile = os.path.join(outputDir, newFilename)
        shutil.copy(srcImgPath, dstFile)
