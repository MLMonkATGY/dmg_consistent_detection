import ujson as json
import glob
from PIL import Image
import shutil

trainAnnPath = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/train.json"
srcImgDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/negative_sample"
outputImgDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/images_neg"
with open(trainAnnPath, "r") as f:
    trainAnn = json.load(f)

allTrainImg = trainAnn["images"]
allTrainAnn = trainAnn["annotations"]
lastImgId = max([x["id"] for x in allTrainImg])
lastAnnId = max([x["id"] for x in allTrainAnn])

search = f"{srcImgDir}/**/*.jpg"
allImgPath = glob.glob(search, recursive=True)
allNewImgAnn = []
allNewAnn = []
for imgId, imgPath in enumerate(allImgPath):
    shutil.copy2(imgPath, outputImgDir)
    im = Image.open(imgPath)
    width, height = im.size
    filename = imgPath.split("/")[-1]
    newImgAnn = {
        "file_name": filename,
        "height": height,
        "id": lastImgId + imgId + 1,
        "width": width,
    }
    allNewImgAnn.append(newImgAnn)


for annId, newImgAnn in enumerate(allNewImgAnn):
    imgId = newImgAnn["id"]
    newAnn = {
        "area": 2,
        "bbox": [0, 0, 1, 1],
        "category_id": 0,
        "id": lastAnnId + annId + 1,
        "ignore": 0,
        "image_id": imgId,
        "iscrowd": 0,
        "segmentation": [],
    }

    allNewAnn.append(newAnn)


allTrainImg.extend(allNewImgAnn)
allTrainAnn.extend(allNewAnn)
trainAnn["images"] = allTrainImg
trainAnn["annotations"] = allTrainAnn
with open("./train_neg.json", "w") as f:
    json.dump(trainAnn, f)
