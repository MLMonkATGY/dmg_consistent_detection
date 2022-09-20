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

from ml_research.train.OODFilterTrainer import GetDataloaders
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import ujson as json


def StageModel(weightPath: str):
    testMetric = MeanAveragePrecision(class_metrics=True)
    expModelPerfFile = "./latest_staging_perf.json"
    origFilename = weightPath.split("/")[-1].split(".")[0]
    runId = stagingModelInfo.runId
    baseOutputDir = "/".join(weightPath.split(os.path.sep)[:-1])
    device = torch.device("cuda")
    with open(weightPath, "rb") as f:
        modelScript = torch.jit.load(f, map_location=device)
    # modelScript = mlflow.pytorch.load_model(
    #     model_uri=f"models:/{stagingModelInfo.taskName}/{2}"
    # )
    _, valLoader = GetDataloaders()
    with torch.no_grad():
        for batch in tqdm(valLoader):
            imgs, targets = batch
            images = list(image.half().to(device) for image in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            _, preds = modelScript(images)
            testMetric.update(preds, targets)
        metricResult = testMetric.compute()
        metricSerializable = {
            k: v.cpu().numpy().tolist() for k, v in metricResult.items()
        }
        with open(expModelPerfFile, "w") as f:
            json.dump(metricSerializable, f)
        pprint(metricResult)

    stagingName = "{0}_torchscript".format(origFilename)

    if metricResult["map_50"] < 0.0:
        raise Exception()
    mlflow.pytorch.log_model(
        pytorch_model=modelScript,
        artifact_path=stagingName,
        registered_model_name=stagingModelInfo.taskName,
        extra_files=[expModelPerfFile],
    )

    mlflow.log_artifacts(baseOutputDir)
