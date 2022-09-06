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


def create_model(num_classes):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        pretrained=True, min_size=300, max_size=600, num_classes=6
    )

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
            # torchvision.transforms.ColorJitter(
            #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            # ),
            torchvision.transforms.ToTensor(),
        ]
    )
    trainDs = CocoDataset(
        root="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/images",
        annotation="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/complete.json",
        transforms=trainTransform,
    )

    trainLoader = DataLoader2(
        trainDs, shuffle=True, batch_size=10, num_workers=4, collate_fn=collate_fn
    )
    num_classes = 6
    model = create_model(num_classes)
    model = model.to(DEVICE)
    model.train()
    metric = MeanAveragePrecision(class_metrics=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in tqdm(range(1000), desc="epoch"):
        model.train()
        for batchId, (imgs, targets) in enumerate(tqdm(trainLoader)):
            optimizer.zero_grad()

            images = list(image.to(DEVICE) for image in imgs)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            # with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            if batchId % 5 == 0:
                tqdm.write(str(losses.detach().cpu().numpy()))
        model.eval()
        if e % 10 != 0:
            continue
        with torch.no_grad():
            for imgs, targets in trainLoader:
                optimizer.zero_grad()

                images = list(image.to(DEVICE) for image in imgs)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                # with autocast():
                preds = model(images)
                # losses = sum(loss for loss in loss_dict.values())
                # print(preds[0])
                metric.update(preds, targets)
        metricResult = metric.compute()
        pprint(metricResult)
        visualizeAll(model=model)
        metric.reset()
