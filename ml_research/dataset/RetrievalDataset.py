import torchvision
from torchvision.datasets import ImageFolder
from typing import Optional, Callable, Any, Tuple
import os
import torch
from torch.utils.data import DataLoader2
from torchvision.models.efficientnet import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
)
import cv2
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.losses import ProxyNCALoss

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import numpy as np
from torch.optim.adam import Adam
from tqdm import tqdm
from torchmetrics import Accuracy, ConfusionMatrix


class RetrievalDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, targets = super().__getitem__(index)
        imgPath = self.samples[index][0]
        caseId = int(imgPath.split(os.path.sep)[-1].split("_")[0])
        caseIdTensor = torch.tensor(caseId, dtype=torch.int64)
        return img, targets, caseIdTensor


if __name__ == "__main__":
    root = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/ex2/train"
    root2 = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/ex2/test"
    targetPG = [10, 20, 30, 110]
    maxEpoch = 500

    trainTransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((350, 350)),
            torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            torchvision.transforms.RandomHorizontalFlip(p=0.2),
            torchvision.transforms.ToTensor(),
        ]
    )
    evalTransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((350, 350)),
            torchvision.transforms.ToTensor(),
        ]
    )
    outFeatSize = 2
    device = torch.device("cuda")
    model = torchvision.models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(
        in_features=num_ftrs, out_features=outFeatSize
    )
    trainAcc = Accuracy(num_classes=outFeatSize).to(device)
    trainConfMat = ConfusionMatrix(num_classes=2, normalize="true").to(device)
    model = model.to(device)
    ds = RetrievalDataset(root, trainTransform)
    ds2 = RetrievalDataset(root2, evalTransform)

    loader = DataLoader2(ds, batch_size=16, num_workers=3, shuffle=True)
    loader2 = DataLoader2(ds2, batch_size=16, num_workers=3, shuffle=False)
    pgCostDf = pd.read_csv(
        "/home/alextay96/Desktop/workspace/dmg_price/data/pg_cost.csv"
    )
    # loss_func = ProxyNCALoss(4, outFeatSize).to(torch.device("cuda"))
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=1e-3)
    # loss_optimizer = Adam(loss_func.parameters(), lr=1e-1)

    allFeatureInOrder = []

    for e in tqdm(range(maxEpoch), desc="Epoch"):
        model.train()
        allImgs = []
        allCaseId = []
        for i in tqdm(loader, desc="train"):
            optimizer.zero_grad()
            # loss_optimizer.zero_grad()
            img, targets, caseIdTensor = i
            img = img.to(device)
            targets = targets.to(device)
            output = model(img)
            losses = loss_func(output, targets)
            preds = torch.argmax(output, dim=1)
            trainAcc.update(preds, targets)
            trainConfMat.update(preds, targets)
            allImgs.extend(img)
            allCaseId.extend(caseIdTensor)
            losses.backward()
            optimizer.step()
            # loss_optimizer.step()
        print(trainAcc.compute())
        print(trainConfMat.compute())

        trainAcc.reset()
        if e % 5 != 0:
            continue
        # featureDs = torch.stack(allFeatureInOrder, dim=0)
        match_finder = MatchFinder(distance=LpDistance())
        inference_model = InferenceModel(model, match_finder=match_finder)
        inference_model.train_knn(allImgs)
        allGt = []
        allPreds = []
        model.eval()
        with torch.no_grad():

            for i in tqdm(loader2, desc="eval"):
                img, targets, gtCaseIdList = i
                img = img.to(device)

                distances, indices = inference_model.get_nearest_neighbors(img, k=1)

                for gtCaseId, predictedIndex in zip(gtCaseIdList, indices):
                    gtCaseId = gtCaseId.numpy()
                    gtCost = pgCostDf[
                        (pgCostDf["CaseID"] == gtCaseId)
                        & (pgCostDf["PartGroup"].isin(targetPG))
                    ]["Cost"].sum()
                    if gtCost < 1000:
                        continue
                    allGt.append(gtCost)
                    allPredictedCost = []
                    for predicted in predictedIndex:
                        nearestCaseId = allCaseId[predicted].numpy()
                        predCost = pgCostDf[
                            (pgCostDf["CaseID"] == nearestCaseId)
                            & (pgCostDf["PartGroup"].isin(targetPG))
                        ]["Cost"].sum()
                        allPredictedCost.append(predCost)
                    avgPredCost = np.mean(allPredictedCost)
                    allPreds.append(avgPredCost)
            evalDf = pd.DataFrame([allGt, allPreds], index=["gt", "preds"]).transpose()
            avgMape = mean_absolute_percentage_error(
                evalDf["gt"].values, evalDf["preds"].values
            )
            avgMae = mean_absolute_error(evalDf["gt"].values, evalDf["preds"].values)
            print(f"MAPE : {avgMape}")
            print(f"MAE : {avgMae}")
            print("a")

    # nearest_imgs = [dataset[i][0] for i in indices.cpu()[0]]
    # print("nearest images")
    # imshow(torchvision.utils.make_grid(nearest_imgs))
    # allFeatureInOrder.extend(output)
