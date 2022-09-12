from pprint import pprint
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from ml_research.dataset.CocoDataset import CocoDataset

# from ml_research.dataset.DataModule import CocoDatamodule
import torch
from torch.utils.data import DataLoader2
from PIL import Image
from pycocotools.coco import COCO
import os
from torch.cuda.amp import autocast
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from tqdm import tqdm
from ml_research.eval.visualize import visualizeAll
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchmetrics


def create_model():

    # load Faster RCNN pre-trained model
    model = torchvision.models.efficientnet_b2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features=num_ftrs, out_features=2)

    # model.fc = torch.nn.Linear(2048, 2)
    # get the number of input features
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # define a new head for the detector with required number of classes
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    DEVICE = torch.device("cuda")
    # dm = CocoDatamodule()
    trainTransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((480, 480)),
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            torchvision.transforms.RandomApply(
                [torchvision.transforms.GaussianBlur(5, sigma=(0.1, 0.2))], p=0.2
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.2),
            torchvision.transforms.ToTensor(),
        ]
    )
    evalTransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((480, 480)),
            torchvision.transforms.ToTensor(),
        ]
    )
    trainDs = ImageFolder(
        root="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/dmg_area/myvi_range_rear/train",
        transform=trainTransform,
    )
    evalDs = ImageFolder(
        root="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/dmg_area/myvi_range_rear/test",
        transform=evalTransform,
    )

    trainLoader = DataLoader2(trainDs, shuffle=True, batch_size=20, num_workers=4)

    evalLoader = DataLoader2(evalDs, shuffle=False, batch_size=32, num_workers=4)
    num_classes = 6
    model = create_model()
    model = model.to(DEVICE)
    model.train()
    trainAcc = torchmetrics.Accuracy(num_classes=2).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    trainConfMat = torchmetrics.ConfusionMatrix(num_classes=2, normalize="true").to(
        DEVICE
    )
    evalAcc = torchmetrics.Accuracy(num_classes=2).to(DEVICE)
    evalConfMat = torchmetrics.ConfusionMatrix(num_classes=2, normalize="true").to(
        DEVICE
    )
    loggingLossInterval = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in tqdm(range(1000), desc="epoch"):
        model.train()
        for batchId, (imgs, targets) in enumerate(tqdm(trainLoader)):
            optimizer.zero_grad()

            images = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            with autocast():
                logit = model(images)
            preds = torch.argmax(logit, dim=1)
            losses = criterion(logit, targets)
            losses.backward()
            optimizer.step()
            trainAcc.update(preds, targets)
            trainConfMat.update(preds, targets)
            if batchId % loggingLossInterval == 0:
                tqdm.write(str(losses.detach().cpu().numpy()))
        print(f"Train Acc : {trainAcc.compute()}")
        pprint(trainConfMat.compute())
        trainConfMat.reset()
        trainAcc.reset()
        model.eval()

        with torch.no_grad():
            for imgs, targets in tqdm(evalLoader):
                optimizer.zero_grad()
                images = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                with autocast():
                    logit = model(images)
                preds = torch.argmax(logit, dim=1)

                evalConfMat.update(preds, targets)
                evalAcc.update(preds, targets)
        metricResult = evalConfMat.compute()
        print(f"Eval Acc : {evalAcc.compute()}")

        evalAcc.reset()
        pprint(metricResult)
        evalConfMat.reset()
