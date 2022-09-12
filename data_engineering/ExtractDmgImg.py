import pandas as pd
import ujson as json

if __name__ == "__main__":
    p = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/dmg_area/src_img.csv"

    df = pd.read_csv(p)
    target = "damage_area"
    targetFile = []
    nonDmgFile = []
    for _, row in df.iterrows():
        label = json.loads(row["label"])
        allCat = [y for x in label for y in x["rectanglelabels"]]
        if target in allCat:
            targetFile.append(row["image"])

    with open("./dmg_file.txt", "w") as f:
        f.write("\n".join(targetFile))
