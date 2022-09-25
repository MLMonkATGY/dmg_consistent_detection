from operator import itemgetter
from pprint import pprint
import mlflow
import os
from mlflow.tracking import MlflowClient
import urllib3
from urllib3.response import HTTPResponse
from minio import Minio
from ml_engineering.StagingModelInfo import stagingModelInfo
import torch

from ml_research.train.OODFilterTrainer import GetDataloaders, GetNegSampleLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import ujson as json
from loguru import logger
import torchvision
from PIL import Image
import glob


def TestNeg(modelScript):
    inputDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/neg_sample_only"
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    allImgPath = glob.glob(f"{inputDir}/**/*.JPG", recursive=True)
    assert len(allImgPath) > 10
    device = torch.device("cuda")
    totalNegSample = len(allImgPath)
    allNonZeroPred = []
    threshold = 0.6
    fp = 0
    for imgPath in tqdm(allImgPath):
        pilImg = Image.open(imgPath)
        input = transform(pilImg)
        inputBatch = [input]
        images = list(image.to(device) for image in inputBatch)
        _, preds = modelScript(images)
        for pred in preds:
            falseP = pred["scores"] > threshold
            fpPerSample = torch.any(falseP)
            fp += 1 if torch.any(fpPerSample) else 0
    falsePRate = fp / totalNegSample
    print(f"FP : {falsePRate}")
    return falsePRate, totalNegSample


def StageModel(weightPath: str):
    testMetric = MeanAveragePrecision(class_metrics=True)
    expModelPerfFile = "./latest_staging_perf.json"
    origFilename = weightPath.split("/")[-1].split(".")[0]
    runId = stagingModelInfo.runId
    baseOutputDir = "/".join(weightPath.split(os.path.sep)[:-1])
    device = torch.device("cuda")
    with open(weightPath, "rb") as f:
        modelScript = torch.jit.load(f, map_location=device)

    trainLoader, valLoader = GetDataloaders()
    falsePRate, totalNegSample = TestNeg(modelScript)

    with torch.no_grad():
        for batch in tqdm(valLoader):
            imgs, targets = batch
            images = list(image.to(device) for image in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            _, preds = modelScript(images)
            testMetric.update(preds, targets)
        metricResult = testMetric.compute()
        metricSerializable = {
            k: v.cpu().numpy().tolist() for k, v in metricResult.items()
        }
        metricSerializable["false_positive_rate"] = falsePRate
        metricSerializable["false_neg_sample_num"] = totalNegSample

        with open(expModelPerfFile, "w") as f:
            json.dump(metricSerializable, f)
        pprint(metricResult)

    stagingName = "{0}_torchscript".format(origFilename)

    # if metricResult["map_50"] < 0.8:
    #     raise Exception()
    mlflow.pytorch.log_model(
        pytorch_model=modelScript,
        artifact_path=stagingName,
        registered_model_name=stagingModelInfo.taskName,
        extra_files=[expModelPerfFile],
    )

    mlflow.log_artifacts(baseOutputDir)
    modelScript = mlflow.pytorch.load_model(
        model_uri=f"models:/{stagingModelInfo.taskName}/{1}"
    )
    localFilename = f"./{stagingModelInfo.taskName}_v{1}.pth"
    with open(localFilename, "wb") as f:
        torch.jit.save(modelScript, f)
    logger.success(f"Saved model in local : {localFilename}")
