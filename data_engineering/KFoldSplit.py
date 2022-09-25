import os
import pandas as pd
import shutil
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from loguru import logger

srcDfPath = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/Saloon-4Dr_FrontView_cls.csv"
outputFilename = "KFold_" + srcDfPath.split("/")[-1].split(".")[0]
annDf = pd.read_csv(srcDfPath)
allViews = annDf["view_name"].unique()
n_splits = 5
imgBaseDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vType_range/Saloon-4Dr_FrontView_cls"

blackListImg = []
nonExistImg = []
for filename in tqdm(annDf["dst_filename"].unique()):
    srcPath = os.path.join(imgBaseDir, filename)
    try:

        if not os.path.exists(srcPath):
            nonExistImg.append(filename)
            continue
        img = Image.open(srcPath)
    except Exception as e1:
        logger.warning(filename)
        nonExistImg.append(filename)
annDf = annDf[~annDf["dst_filename"].isin(nonExistImg)]
print(
    annDf.groupby(["view_name", "label"])["dst_filename"]
    .count()
    .sort_values(ascending=False)
)

for viewName in allViews:
    viewSample = annDf[annDf["view_name"] == viewName]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    x = viewSample
    y = viewSample["label"]
    allFolds = []
    for foldId, (train_index, test_index) in enumerate(skf.split(x, y)):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        # y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train["train_test"] = "train"
        X_test["train_test"] = "test"

        foldDf = pd.concat([X_train, X_test]).sort_index(ascending=True)
        foldDf["kfold"] = foldId + 1
        print(foldDf.head(30))
        allFolds.append(foldDf)
    allFoldDf = pd.concat(allFolds)
    allFoldDf.to_csv(f"./{outputFilename}_kfold_{n_splits}_{viewName}.csv")
