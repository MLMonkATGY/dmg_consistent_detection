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
    # OODCsvPath = loadAndFilter()
    OODCsvPath = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/OOD_reject/Hatchback-5Dr_FrontView_cls/rej_Hatchback-5Dr_FrontView_cls.csv"
    dataLoaderGenerator = KFoldDatasetGenerator(OODCsvPath)
    allLoader = dataLoaderGenerator.genDataloader()
    allKFoldRunId = []
    maxIterations = 3
    for iteration in range(1, maxIterations + 1):
        for trainLoader, testLoader in allLoader:
            runId = trainKthFold(trainLoader, testLoader, iteration)
            allKFoldRunId.append(runId)
        predCsv = CrossValPred(iteration, dataLoaderGenerator)
        LabelIssueRankFilter(predCsv, iteration)
