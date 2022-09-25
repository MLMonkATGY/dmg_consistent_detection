from re import T
import pandas as pd
import mlflow
from torchvision import transforms
import glob
from PIL import Image
import torch
from tqdm import tqdm
from ml_research.params import ImportEnv

if __name__ == "__main__":
    device = torch.device("cuda")
    OODModel = mlflow.pytorch.load_model(model_uri=f"models:/mrm_OOD_detector/None")
    OODModel = OODModel.to(device)
    OODModel.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    inputDir = "/home/alextay96/Desktop/workspace/extract_img"
    search = f"{inputDir}/**/*.JPG"
    allImgs = glob.glob(search, recursive=True)
    batchSize = 20
    threshold = 0.6
    imgBatch = []
    output = []
    for i in range(0, len(allImgs), batchSize):
        imgBatch.append(allImgs[i : i + batchSize])

    with torch.no_grad():
        for batch in tqdm(imgBatch):
            inputList = []
            for path in batch:
                pilImg = Image.open(path)
                inputTensor = transform(pilImg)
                inputList.append(inputTensor.to(device))
            _, preds = OODModel(inputList)
            for pred, path in zip(preds, batch):
                falseP = pred["scores"] > threshold
                detection = torch.any(falseP)
                payload = {"filename": path.split("/")[-1], "isOOD": False}

                if torch.any(detection):
                    payload["isOOD"] = False
                else:
                    payload["isOOD"] = True
                output.append(payload)
    df = pd.json_normalize(output)
    df.to_csv("./OOD_filter_final.csv")
