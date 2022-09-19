import seaborn as sns
import cv2
import pandas as pd
import ujson as json
from datetime import datetime, timedelta
import random
import numpy as np


def daterange(start_date, end_date):
    delta = timedelta(hours=1)
    allDate = []
    while start_date < end_date:
        allDate.append(start_date)
        start_date += delta
    return allDate


start_date = datetime(2022, 9, 12, 23, 00)
end_date = datetime(2022, 9, 20, 23, 00)
allDate = daterange(start_date, end_date)
allSensorCount = {f"sensor_{x}": [] for x in range(9)}
for i in allDate:
    for j in range(9):
        allSensorCount[f"sensor_{j}"].append(random.randint(0, 100))
mockDf = pd.DataFrame(allSensorCount)
mockDf["datetime"] = allDate
print(mockDf.head(30))
maxVal = 100

img = cv2.imread(
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/Picture1.png"
)
annotDf = pd.read_csv(
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/random_patch.csv"
)
color = (255, 0, 0)
thickness = -1
with open(
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/a.json",
    "r",
) as f:
    ann = json.load(f)

labelStr = annotDf["label"].item()
allLabels = json.loads(labelStr)
for label in ann["annotations"]:
    x = label["bbox"][0]
    y = label["bbox"][1]
    w = label["bbox"][2]
    h = label["bbox"][3]
    startP = (x, y)
    endP = (x + w, y + h)
    sub_img = img[y : y + h, x : x + w]

    white_rect = np.zeros(sub_img[:, :, 0].shape, dtype=np.uint8) + 10
    im_color = cv2.applyColorMap(white_rect, cv2.COLORMAP_JET)
    res = cv2.addWeighted(sub_img, 0.5, im_color, 0.5, 1.0)

    img[y : y + h, x : x + w] = res


cv2.imshow("Heatmap", img)
key = cv2.waitKey(0)
