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
from ml_research.dataset.KFoldDatasetGenerator import KFoldDatasetGenerator
from ml_research.params.PriceRangeParams import PriceRangeParams
from pytorch_lightning.callbacks import ModelCheckpoint
from ml_research.params import ImportEnv
from loguru import logger as displayLogger
from mlflow.tracking.client import MlflowClient
from colorama import Fore
import itertools
import warnings
import torchvision
import torchmetrics
import dataclasses

warnings.filterwarnings("ignore")

# obj_list = ["damage", "not_damaged"]


torch.manual_seed(PriceRangeParams.randomSeed)


@dataclass
class SampleEvalResult:
    pred: Any
    originalImgs: Any
    scales: Any


class ProcessModel(pl.LightningModule):
    def __init__(
        self,
    ):
        super(ProcessModel, self).__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.efficientnet_b2(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features=num_ftrs, out_features=2)

        self.isHparamLogged = False
        self.criterion = torch.nn.CrossEntropyLoss()
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

        self.log("e_acc", evalAcc, prog_bar=True)
        self.log("e_0_TP", class0TP, prog_bar=True)
        self.log("eval_class0_FN", class0FN)
        self.log("eval_class1_FN", class1FN)
        self.log("e_1_TP", class1TP, prog_bar=True)

        self.evalConfMatMetric.reset()
        self.evalAccMetric.reset()
        return super().on_validation_epoch_end()


def trainKthFold(trainLoader, testLoader, foldId):
    logger = MLFlowLogger(
        experiment_name=PriceRangeParams.experimentName,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="e_acc",
        save_top_k=PriceRangeParams.saveTopNBest,
        mode="max",
        filename="{epoch:03d}-{e_acc:.2f}-{e_0_TP:.2f}-{e_1_TP:.2f}",
    )
    trainProcessModel = ProcessModel()
    trainer = pl.Trainer(
        default_root_dir="./outputs/{}".format(PriceRangeParams.localSaveDir),
        max_epochs=PriceRangeParams.maxEpoch,
        gpus=1,
        check_val_every_n_epoch=PriceRangeParams.check_val_every_n_epoch,
        num_sanity_val_steps=3,
        benchmark=True,
        precision=PriceRangeParams.trainingPrecision,
        logger=logger,
        log_every_n_steps=20,
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


if __name__ == "__main__":
    dataLoaderGenerator = KFoldDatasetGenerator()
    allLoader = dataLoaderGenerator.genDataloader()
    for foldId, (trainLoader, testLoader) in enumerate(allLoader):
        trainKthFold(trainLoader, testLoader, foldId)
