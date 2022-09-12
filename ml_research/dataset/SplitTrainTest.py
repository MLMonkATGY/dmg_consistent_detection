from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from collections import Counter
import os
import shutil
import glob
import random
from tqdm import tqdm

if __name__ == "__main__":
    root = "/home/alextay96/Desktop/workspace/scrape_mrm/saloon_4_dr_range/Rear_View"
    ds = ImageFolder(root=root)
    counter = Counter(ds.targets)
    print(counter)
    minSample = min(counter.values())
    outputDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/dmg_area/myvi_range_rear"
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.makedirs(outputDir, exist_ok=True)
    usedFile = set()
    for split in ["train", "test"]:
        splitDir = os.path.join(outputDir, split)
        os.makedirs(splitDir, exist_ok=True)
        for className in ["0", "1"]:
            classSrcDir = os.path.join(root, className)
            imgInClass = glob.glob(f"{classSrcDir}/**/*.jpg", recursive=True)
            imgInClass = [x for x in imgInClass if x not in usedFile]
            random.shuffle(imgInClass)
            sample = imgInClass[: minSample // 2]
            destDirName = os.path.join(splitDir, className)
            os.makedirs(destDirName, exist_ok=True)
            for srcFile in tqdm(sample):
                shutil.copy(srcFile, destDirName)
                usedFile.add(srcFile)
