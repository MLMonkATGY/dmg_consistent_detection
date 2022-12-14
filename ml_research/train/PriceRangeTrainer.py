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

warnings.filterwarnings("ignore")

# obj_list = ["damage", "not_damaged"]


torch.manual_seed(PriceRangeParams.randomSeed)


@dataclass
class SampleEvalResult:
    pred: Any
    originalImgs: Any
    scales: Any


def GenerateRunName(foldId, iteration):
    imgBaseDir = PriceRangeParams.imgBaseDir
    clsName = imgBaseDir.split("/")[-1].replace("_cls", "")
    runName = f"{clsName}_i{iteration}_k{foldId}"
    return runName


def transportBestModel(dirPath, foldId):
    baseOutput = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/auto_select"
    if os.path.exists(baseOutput) and foldId == 1:
        shutil.rmtree(baseOutput)
    os.makedirs(baseOutput, exist_ok=True)
    foldOutputDir = os.path.join(baseOutput, str(foldId))
    os.makedirs(foldOutputDir, exist_ok=True)
    allSavedModel = os.listdir(dirPath)
    latestEpoch = 0
    targetModel = ""
    for p in allSavedModel:
        epoch = int(p.split("epoch=")[-1].split("-e_acc")[0])
        if epoch > latestEpoch:
            latestEpoch = epoch
            targetModel = p
    shutil.copy(os.path.join(dirPath, targetModel), foldOutputDir)


class ProcessModel(pl.LightningModule):
    def __init__(self, ceWeight=[1.0, 1.0]):
        super(ProcessModel, self).__init__()
        self.save_hyperparameters()
        print(f"CE Weight {ceWeight}")
        self.model = torchvision.models.efficientnet_b0(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features=num_ftrs, out_features=2)

        self.isHparamLogged = False
        classWeight = torch.tensor(ceWeight)
        self.criterion = torch.nn.CrossEntropyLoss(classWeight)
        self.evalAccMetric = torchmetrics.Accuracy(num_classes=2)
        self.trainAccMetric = torchmetrics.Accuracy(num_classes=2)
        self.trainConfMatMetric = torchmetrics.ConfusionMatrix(
            num_classes=2, normalize="true"
        )
        self.evalConfMatMetric = torchmetrics.ConfusionMatrix(
            num_classes=2, normalize="true"
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), PriceRangeParams.learningRate)

    def forward(self, imgs):
        logit = self.model(imgs)
        return logit

    def training_step(self, batch, batch_idx):
        imgs, annot = batch

        output = self(imgs)
        loss = self.criterion(output, annot)
        preds = torch.argmax(output, dim=1)
        self.trainAccMetric.update(preds, annot)
        self.trainConfMatMetric.update(preds, annot)
        self.log("ce_loss", loss)

        return loss

    def training_epoch_end(self, output):
        confMat = self.trainConfMatMetric.compute()
        trainAcc = self.trainAccMetric.compute()
        class0TP = confMat[0][0]
        class0FN = confMat[0][1]
        class1FN = confMat[1][0]
        class1TP = confMat[1][1]

        self.log("t_acc", trainAcc, prog_bar=True)
        self.log("train_class0_TP", class0TP)
        self.log("train_class0_FN", class0FN)
        self.log("train_class1_FN", class1FN)
        self.log("train_class1_TP", class1TP)
        self.trainConfMatMetric.reset()
        self.trainAccMetric.reset()

    def validation_step(self, batch, idx):
        imgs, annot = batch
        output = self(imgs)
        preds = torch.argmax(output, dim=1)
        self.evalAccMetric.update(preds, annot)
        self.evalConfMatMetric.update(preds, annot)

    def validation_epoch_end(self, val_step_outputs) -> None:
        confMat = self.evalConfMatMetric.compute()
        evalAcc = self.evalAccMetric.compute()
        class0TP = confMat[0][0]
        class0FN = confMat[0][1]
        class1FN = confMat[1][0]
        class1TP = confMat[1][1]
        diffTP = torch.abs(class0TP - class1TP)
        self.log("e_acc", evalAcc, prog_bar=True)
        self.log("e_0_TP", class0TP, prog_bar=True)
        self.log("eval_class0_FN", class0FN)
        self.log("eval_class1_FN", class1FN)
        self.log("e_1_TP", class1TP, prog_bar=True)
        self.log("diff_tp", diffTP, prog_bar=True)

        self.evalConfMatMetric.reset()
        self.evalAccMetric.reset()
        return super().on_validation_epoch_end()


def trainKthFold(trainLoader, testLoader, iteration, ceWeight):
    foldId = testLoader.dataset.df["kfold"].unique().item()
    runName = GenerateRunName(foldId, iteration)
    logger = MLFlowLogger(
        experiment_name=PriceRangeParams.experimentName,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        run_name=runName,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="diff_tp",
        save_top_k=PriceRangeParams.saveTopNBest,
        mode="min",
        filename="{epoch:02d}-{e_acc:.2f}-{e_0_TP:.2f}-{e_1_TP:.2f}",
    )
    trainProcessModel = ProcessModel(ceWeight)
    trainer = pl.Trainer(
        default_root_dir="./outputs/{}".format(PriceRangeParams.localSaveDir),
        max_epochs=PriceRangeParams.maxEpoch,
        gpus=1,
        check_val_every_n_epoch=PriceRangeParams.check_val_every_n_epoch,
        num_sanity_val_steps=3,
        benchmark=True,
        precision=PriceRangeParams.trainingPrecision,
        logger=logger,
        log_every_n_steps=40,
        callbacks=[checkpoint_callback],
        detect_anomaly=False,
    )

    trainer.fit(
        trainProcessModel, train_dataloaders=trainLoader, val_dataloaders=testLoader
    )
    displayLogger.success("Started Uploading Best Checkpoints..")
    mlflowLogger: MlflowClient = trainProcessModel.logger.experiment
    mlflowLogger.log_dict(
        trainProcessModel.logger.run_id,
        dataclasses.asdict(PriceRangeParams),
        "hyperparams.json",
    )
    mlflowLogger.log_param(trainProcessModel.logger.run_id, "kth_fold", foldId)

    mlflowLogger.log_artifacts(
        trainProcessModel.logger.run_id,
        checkpoint_callback.dirpath.replace("/checkpoints", "/"),
    )
    transportBestModel(checkpoint_callback.dirpath, foldId)
    return trainProcessModel.logger.run_id
