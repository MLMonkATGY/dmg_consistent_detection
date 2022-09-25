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
from ml_research.params.OODDetectorParams import OODParams
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
from ml_research.dataset.CocoDataset import CocoDataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights


def GenerateRunName(foldId, iteration):
    imgBaseDir = OODParams.imgBaseDir
    clsName = imgBaseDir.split("/")[-1].replace("_cls", "")
    runName = f"{clsName}_i{iteration}_k{foldId}"
    return runName


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        pretrain=True,
        min_size=OODParams.imgMinSize,
        max_size=OODParams.imgMaxSize,
        num_classes=num_classes,
    )
    return model


def GetDataloaders():
    trainTransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            torchvision.transforms.ToTensor(),
        ]
    )
    evalTransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    trainDs = CocoDataset(
        root=OODParams.imgDir,
        annotation=OODParams.trainAnnFile,
        transforms=trainTransform,
    )

    trainLoader = DataLoader2(
        trainDs,
        shuffle=True,
        batch_size=OODParams.trainBatchSize,
        num_workers=OODParams.trainCPUWorker,
        collate_fn=collate_fn,
    )
    evalDs = CocoDataset(
        root=OODParams.imgDir,
        annotation=OODParams.evalAnnFile,
        transforms=evalTransform,
    )
    evalLoader = DataLoader2(
        evalDs,
        shuffle=False,
        batch_size=OODParams.trainBatchSize,
        num_workers=2,
        collate_fn=collate_fn,
    )
    return trainLoader, evalLoader


def GetNegSampleLoader():

    evalTransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    negSampleAnn = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/negative_sample_test.json"

    evalDs = CocoDataset(
        root=OODParams.imgDir,
        annotation=negSampleAnn,
        transforms=evalTransform,
    )
    evalLoader = DataLoader2(
        evalDs,
        shuffle=False,
        batch_size=OODParams.trainBatchSize,
        num_workers=2,
        collate_fn=collate_fn,
    )
    return evalLoader


class ProcessModel(pl.LightningModule):
    def __init__(self):
        super(ProcessModel, self).__init__()
        self.model = create_model(5)
        self.testMetric = MeanAveragePrecision(class_metrics=True)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), OODParams.learningRate)

    def forward(self, imgs, targets=None):
        logit = self.model(imgs, targets)
        return logit

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        images = list(image.to(self.device) for image in imgs)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        [self.log(k, loss) for k, loss in loss_dict.items()]
        self.log("train_loss", losses)
        return losses

    def validation_step(self, batch, idx):
        imgs, targets = batch
        # device = torch.device("cuda")
        images = list(image.to(self.device) for image in imgs)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        preds = self.model(images)
        self.testMetric.update(preds, targets)

    def validation_epoch_end(self, val_step_outputs) -> None:
        try:
            metricResult = self.testMetric.compute()
            for k, v in metricResult.items():
                if "per_class" in k:
                    for i in range(len(v)):
                        clsname = f"{k}_c{i}"
                        self.log(clsname, v[i])
                elif k == "map_50" or k == "mar_100":
                    self.log(k, v, prog_bar=True)
                else:
                    self.log(k, v)
        except Exception as e1:
            logger.warning("Eval metric not updated due to exceptions")
            logger.warning(e1)
        self.testMetric.reset()
        return super().on_validation_epoch_end()


def trainEval():
    runName = "OOD_Object_detection"
    trainLoader, valLoader = GetDataloaders()
    logger = MLFlowLogger(
        experiment_name=OODParams.experimentName,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        run_name=runName,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="map_50",
        save_top_k=OODParams.saveTopNBest,
        mode="max",
        filename="{epoch:02d}-{map_50:.2f}-{mar_100:.2f}",
    )
    trainProcessModel = ProcessModel()
    trainer = pl.Trainer(
        # accumulate_grad_batches=5,
        default_root_dir="./outputs/{}".format(OODParams.localSaveDir),
        max_epochs=OODParams.maxEpoch,
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=OODParams.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        benchmark=True,
        precision=OODParams.trainingPrecision,
        logger=logger,
        log_every_n_steps=200,
        callbacks=[checkpoint_callback],
        detect_anomaly=False,
        # limit_train_batches=10,
        # limit_val_batches=5,
    )

    trainer.fit(
        trainProcessModel, train_dataloaders=trainLoader, val_dataloaders=valLoader
    )
    displayLogger.success("Started Uploading Best Checkpoints..")
    mlflowLogger: MlflowClient = trainProcessModel.logger.experiment
    mlflowLogger.log_dict(
        trainProcessModel.logger.run_id,
        dataclasses.asdict(OODParams),
        "hyperparams.json",
    )

    mlflowLogger.log_artifacts(
        trainProcessModel.logger.run_id,
        checkpoint_callback.dirpath.replace("/checkpoints", "/"),
    )
    return trainProcessModel.logger.run_id


if __name__ == "__main__":
    trainEval()
