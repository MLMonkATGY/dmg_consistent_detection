from pprint import pprint
import ujson as json
from collections import Counter

if __name__ == "__main__":
    annFile = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/ood_2.json"
    with open(annFile, "r") as f:
        ann = json.load(f)
    allBbox = ann["annotations"]
    catByName = {x["id"]: x["name"] for x in ann["categories"]}
    pprint(catByName)
    allClass = [x["category_id"] for x in allBbox]
    counter = Counter(allClass)
    pprint(counter)
