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
from loguru import logger
from ml_research.train.OODFilterTrainer import ProcessModel


def CheckFalsePositive(weightPath):

    device = torch.device("cuda")
    model = ProcessModel.load_from_checkpoint(weightPath)
    model = model.to(device)
    model.eval()
    _, valLoader = GetDataloaders()
    fp = 0
    total_img = 0
    with torch.no_grad():
        for batch in tqdm(valLoader):
            imgs, targets = batch
            images = list(image.to(device) for image in imgs)
            preds = model(images)

            total_img += len(images)
            for p in preds:
                wrong = torch.sum(p["scores"] > 0.6)
                if wrong != 0:
                    fp += 1
    print(f"Total img : {total_img}")
    print(f"Total fp : {fp}")
    print(f"Total fp fraction : {fp / total_img}")


if __name__ == "__main__":
    modelWeight = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/outputs/mlruns/56/bcaaaae979c64a18a9bb231156d1396f/checkpoints/epoch=11-map_50=0.29-mar_100=0.59.ckpt"
    annPath = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/negative_sample_test.json"
    _, valLoader = GetDataloaders()
    CheckFalsePositive(modelWeight)
