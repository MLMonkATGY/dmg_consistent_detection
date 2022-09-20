from dataclasses import dataclass
from pickle import TRUE
from pprint import pprint
from typing import Any, List
from matplotlib import pyplot
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import albumentations as A

from torch.utils.data import DataLoader2
from torchvision.transforms import transforms
from loguru import logger
from torchvision.datasets import ImageFolder
from torch.cuda.amp.autocast_mode import autocast
from pycocotools.cocoeval import COCOeval

import os
import ujson as json
from tqdm import tqdm
from pycocotools.coco import COCO
import cv2
import numpy as np
import copy
from pytorch_lightning.loggers import MLFlowLogger
from ml_research.analysis.CrossValPred import CrossValPred
from ml_research.analysis.LabelIssueRank import LabelIssueRankFilter
from ml_research.dataset.KFoldDatasetGenerator import KFoldDatasetGenerator
from ml_research.eval.OODFilter import loadAndFilter
from ml_research.params.PriceRangeParams import PriceRangeParams
from pytorch_lightning.callbacks import ModelCheckpoint
from ml_research.params import ImportEnv
from loguru import logger as displayLogger
from mlflow.tracking.client import MlflowClient
import itertools
import warnings
import torchvision
import torchmetrics
import dataclasses
import shutil

from ml_research.train.PriceRangeTrainer import trainKthFold


if __name__ == "__main__":
    OODCsvPath = loadAndFilter()
    allKFoldRunId = []
    maxIterations = PriceRangeParams.filterIteration
    for iteration in range(1, maxIterations + 2):
        dataLoaderGenerator = KFoldDatasetGenerator(OODCsvPath)
        allLoader = dataLoaderGenerator.genDataloader()
        for trainLoader, testLoader in allLoader:
            srcDf = trainLoader.dataset.df
            foldId = trainLoader.dataset.df["kfold"].unique().item()

            clsNum1 = len(
                srcDf[
                    (srcDf["label"] == 1)
                    & (srcDf["train_test"] == "train")
                    & (srcDf["kfold"] == foldId)
                ]
            )
            clsNum0 = len(
                srcDf[
                    (srcDf["label"] == 0)
                    & (srcDf["train_test"] == "train")
                    & (srcDf["kfold"] == foldId)
                ]
            )
            ceWeight = [1.0, clsNum0 / clsNum1]
            runId = trainKthFold(trainLoader, testLoader, iteration, ceWeight)
            allKFoldRunId.append(runId)
        predCsv = CrossValPred(iteration, dataLoaderGenerator)
        LabelIssueRankFilter(predCsv, iteration)
