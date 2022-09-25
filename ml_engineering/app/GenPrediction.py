import pandas as pd
from re import T
import pandas as pd
import mlflow
from torchvision import transforms
import glob
from PIL import Image
import torch
from tqdm import tqdm
from ml_research.params import ImportEnv


def ModelRoute():
    # mrm_price_classifier_MPV_FrontView_cls
    search = f"/home/alextay96/Desktop/workspace/extract_img/**/*View/"
    allDsDir = glob.glob(search, recursive=True)
    router = {}
    for i in tqdm(allDsDir):
        elem = i.split("/")
        dsName = elem[-3]
        viewName = elem[-2]
        modelName = f"mrm_price_classifier_{dsName}_{viewName}_cls"
        # OODModel = mlflow.pytorch.load_model(model_uri=f"models:/{modelName}/None")
        router["/".join(elem[-3:-1])] = modelName
    return router
    print(allDsDir)


if __name__ == "__main__":
    # ModelRoute()
    device = torch.device("cuda")
    # OODModel = OODModel.to(device)
    # OODModel.eval()
    transform = transforms.Compose(
        [transforms.Resize((350, 350)), transforms.ToTensor()]
    )
    gtLabelDf = pd.read_csv(
        "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/ml_engineering/app/pg_cost_sim_1.0.csv"
    )
    search = f"/home/alextay96/Desktop/workspace/extract_img/**/*View/"
    allDsDir = glob.glob(search, recursive=True)
    router = {}
    allPreds = []
    for i in tqdm(allDsDir):
        elem = i.split("/")
        dsName = elem[-3]
        viewName = elem[-2]
        modelName = f"mrm_price_classifier_{dsName}_{viewName}_cls"
        priceModel = mlflow.pytorch.load_model(model_uri=f"models:/{modelName}/None")
        priceModel.eval()
        searchImg = f"{i}/**/*.JPG"
        dsImgs = glob.glob(searchImg, recursive=True)
        batchSize = 32
        imgBatch = []
        for i in range(0, len(dsImgs), batchSize):
            imgBatch.append(dsImgs[i : i + batchSize])
        with torch.no_grad():
            for batch in tqdm(imgBatch):
                inputList = []
                for path in batch:
                    pilImg = Image.open(path)
                    inputTensor = transform(pilImg)
                    inputList.append(inputTensor.half().to(device))
                inputTensor = torch.stack(inputList)
                output = priceModel(inputTensor)
                preds = torch.argmax(output, dim=1)
                for pred, path in zip(preds, batch):
                    caseId = int(path.split("/")[-1].split("_")[0])
                    gtLabelRow = gtLabelDf[gtLabelDf["CaseID"] == caseId]
                    if gtLabelRow.empty:
                        print("No gt label found")
                        gtLabel = -1
                    else:
                        if "FrontView" in path:
                            gtLabel = gtLabelRow["cost_pg_front"].item()
                        else:
                            gtLabel = gtLabelRow["cost_pg_rear"].item()

                    payload = {
                        "filename": path.split("/")[-1],
                        "pred": pred.cpu().numpy().item(),
                        "dsName": dsName,
                        "viewName": viewName,
                        "CaseID": caseId,
                        "gtLabel": gtLabel,
                    }
                    allPreds.append(payload)
    df = pd.json_normalize(allPreds)
    df.to_csv("./img_price_pred.csv")
    # inputDir = "/home/alextay96/Desktop/workspace/extract_img"
    # search = f"{inputDir}/**/*.JPG"
    # allImgs = glob.glob(search, recursive=True)
    # batchSize = 20
    # threshold = 0.6
    # imgBatch = []
    # output = []
    # for i in range(0, len(allImgs), batchSize):
    #     imgBatch.append(allImgs[i : i + batchSize])
