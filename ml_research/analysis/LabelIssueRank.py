from logging import raiseExceptions
from cleanlab.filter import find_label_issues
import pandas as pd
import numpy as np
import ujson as json
import os
import shutil
import cv2


def LabelIssueRankFilter(predCsvPath, iteration):
    predsDf = pd.read_csv(predCsvPath)
    # Only filter high cost
    predsDf = predsDf[predsDf["label"] == 1]
    srcImgDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vType_range/Hatchback-5Dr_FrontView_cls"
    iter = iteration
    removeData = 50
    dsName = srcImgDir.split("/")[-1]
    outputDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/wrong_label"
    print(len(predsDf))
    allLabel = predsDf["label"].values
    allLogit = []
    for logit in predsDf["logit"]:
        aaa = logit.replace("'", "")
        arr = np.array(json.loads(logit))
        # arr = np.expand_dims(arr, 0)
        allLogit.append(arr)
    allLogitNp = np.concatenate([allLogit], axis=0)
    # print(predsDf["label"])

    ranked_label_issues = find_label_issues(
        allLabel,
        allLogitNp,
        return_indices_ranked_by="self_confidence",
        min_examples_per_class=50,
    )

    print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
    # print(f"Top 100 most likely label errors: \n {ranked_label_issues[:removeData]}")

    labelError = ranked_label_issues[:removeData]
    classname = ["gt=low_cost", "gt=high_cost"]
    allDataToRemove = []
    allLabelToRemove = []
    for id, idx in enumerate(labelError):
        row = predsDf.iloc[idx]
        # print(predsDf[511:512].head())
        label = row["label"].item()

        srcImg = os.path.join(srcImgDir, row["dst_filename"])
        # img = cv2.imread(srcImg)
        selection = 114

        # if id < len(labelError) // 2:
        #     selection = 114
        # else:
        # cv2.imshow(classname[label], img)
        # selection = cv2.waitKey(0)
        if selection == 113:
            print("not outlier")
            nameToRemove = row["dst_filename"]
            # predsDf = predsDf[~(predsDf["dst_filename"] == nameToRemove)]
            # print(predsDf[511:512].head())

        elif selection == 114:
            print("outlier")
            allDataToRemove.append(row["dst_filename"])
            allLabelToRemove.append(row["label"])
        cv2.destroyAllWindows()
    wrongLabelDf = pd.DataFrame(allDataToRemove, columns=["dst_filename"])
    wrongLabelDf["orig_label"] = allLabelToRemove
    outputIterDir = f"{outputDir}/{dsName}/iter{iter}"
    if os.path.exists(outputIterDir):
        shutil.rmtree(outputIterDir)
    os.makedirs(outputIterDir, exist_ok=True)

    wrongLabelDf.to_csv(f"{outputIterDir}/wrong_label.csv")
