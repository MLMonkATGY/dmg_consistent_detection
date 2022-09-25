from operator import itemgetter
from pprint import pprint
import mlflow
import os
from mlflow.tracking import MlflowClient
import urllib3
from urllib3.response import HTTPResponse
from minio import Minio
from ml_engineering.StagingModelInfo import StagingModelInfo, stagingModelInfo
import torch

from ml_research.train.OODFilterTrainer import GetDataloaders, GetNegSampleLoader
from torchmetrics import Accuracy, ConfusionMatrix
from tqdm import tqdm
import ujson as json
from loguru import logger
import torchvision
from PIL import Image
import glob
from ml_research.params.PriceRangeParams import PriceRangeParams
import pandas as pd
import glob
import numpy as np
from torch.cuda.amp import autocast


def StagePriceRangeClassifier(weightPath: str):
    baseOutputDir = "/".join(weightPath.split(os.path.sep)[:-1])
    expModelPerfFile = "./price_classifier_range.json"
    stagingName = stagingModelInfo.customEvalImgDir.split("/")[-1]
    kfoldName = 5
    allSrcImgLabelsDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/kfold_src"
    alllabelDfPath = glob.glob(f"{allSrcImgLabelsDir}/**/*.csv", recursive=True)
    allLabelDf = [pd.read_csv(x) for x in alllabelDfPath]
    completeLabelDf = pd.concat(allLabelDf)
    testFilename = completeLabelDf[
        (completeLabelDf["kfold"] == kfoldName)
        & (completeLabelDf["train_test"] == "test")
    ]["dst_filename"].tolist()
    evalTransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (PriceRangeParams.imgMaxSize, PriceRangeParams.imgMaxSize)
            ),
            torchvision.transforms.ToTensor(),
        ]
    )
    device = torch.device("cuda")

    rejectDf = pd.read_csv(
        "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/all_rejects.csv"
    )
    rejectFilename = rejectDf["rej_filename"].tolist()
    accMetric = Accuracy(num_classes=2).to(device)
    confMatMetric = ConfusionMatrix(num_classes=2, normalize="true").to(device)
    allImg = glob.glob(f"{stagingModelInfo.customEvalImgDir}/**/*.JPG", recursive=True)
    before = len(allImg)
    allValdImg = [
        x
        for x in allImg
        if x.split("/")[-1] not in rejectFilename and x.split("/")[-1] in testFilename
    ]
    after = len(allValdImg)
    assert before > after
    with open(weightPath, "rb") as f:
        modelScript = torch.jit.load(f, map_location=device)

    with torch.no_grad():
        for imgPath in tqdm(allValdImg):
            filename = imgPath.split("/")[-1]
            pilImg = Image.open(imgPath)
            imgTensor = evalTransform(pilImg)
            imgTensor = imgTensor.unsqueeze(0)
            imgTensor = imgTensor.half().to(device)
            with autocast():
                output = modelScript(imgTensor)
            predLabel = torch.argmax(output, dim=1)
            gtLabel = completeLabelDf[completeLabelDf["dst_filename"] == filename].iloc[
                0
            ]["label"]
            gtLabelTensor = torch.from_numpy(np.array([gtLabel])).to(device)
            accMetric.update(predLabel, gtLabelTensor)
            confMatMetric.update(predLabel, gtLabelTensor)
    acc = accMetric.compute()
    confMat = confMatMetric.compute()
    pprint(acc)
    pprint(confMat)
    if stagingModelInfo.dryRun:
        logger.success("Dry run mode detected. Skip actual model staging")
        return
    perf = {"acc": acc.cpu().numpy().item()}
    for i, row in enumerate(confMat):
        for j, col in enumerate(row):
            perf[f"conf_mat_row{i}_col{j}"] = col.cpu().numpy().item()

    with open(expModelPerfFile, "w") as f:
        json.dump(perf, f)
    input("Is performance of stage model same as performance in exp model ?")
    logger.success("Performance parity check done. Proceeding to stage model")
    registredModelName = f"{stagingModelInfo.taskName}_{stagingName}"
    logger.success(f"Registered model name : {registredModelName}")

    loggedModel = mlflow.pytorch.log_model(
        pytorch_model=modelScript,
        artifact_path=stagingName,
        registered_model_name=registredModelName,
        extra_files=[expModelPerfFile],
    )

    mlflow.log_artifacts(baseOutputDir)
    logger.success("Done staging model")

    modelScript = mlflow.pytorch.load_model(
        model_uri=f"models:/{registredModelName}/None"
    )
    logger.success("Successfully refetch model from registry")


# StagePriceRangeClassifier()
