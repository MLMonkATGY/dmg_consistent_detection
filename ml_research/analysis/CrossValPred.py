import os
from pprint import pprint
import torch
from ml_research.train.PriceRangeTrainer import ProcessModel
import pandas as pd
import glob
from torch.cuda.amp.autocast_mode import autocast
import torchmetrics
from tqdm import tqdm
import numpy as np


def CrossValPred(iteration, datasetGenerator):
    gen = datasetGenerator
    iter = f"iter_{iteration}"
    taskName = gen.srcDfPath.split("/")[-1]
    allFoldsLoader = gen.genDataloader()
    trainedModelDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/auto_select"
    outputDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/cross_val_pred"
    taskName = iter + "_" + taskName
    device = torch.device("cuda")
    allDfWithPreds = []
    for _, valLoader in allFoldsLoader:
        srcDf: pd.DataFrame = valLoader.dataset.df
        allPredLogit = []
        foldId = srcDf["kfold"].unique().item()
        search = f"{trainedModelDir}/{foldId}/**/*.ckpt"
        accMetrics = torchmetrics.Accuracy(num_classes=2).to(device)
        confMatMetrics = torchmetrics.ConfusionMatrix(
            num_classes=2, normalize="true"
        ).to(device)
        allmodelPath = glob.glob(search, recursive=True)
        modelPath = allmodelPath[0]
        trainedModel = ProcessModel.load_from_checkpoint(modelPath)
        trainedModel = trainedModel.to(device)
        trainedModel.eval()
        with torch.no_grad():
            for img, targets in tqdm(valLoader):
                img = img.to(device)
                targets = targets.to(device)
                with autocast():
                    logit = trainedModel(img)
                preds = torch.argmax(logit, dim=1)
                logitNp = logit.cpu().numpy().tolist()
                allPredLogit.extend(logitNp)
                accMetrics.update(preds, targets)
                confMatMetrics.update(preds, targets)
        assert len(allPredLogit) == len(srcDf)
        srcDf["logit"] = allPredLogit
        srcDf["model_version"] = modelPath
        allDfWithPreds.append(srcDf)
        print(accMetrics.compute())
        pprint(confMatMetrics.compute())
        accMetrics.reset()
        confMatMetrics.reset()
    allDf = pd.concat(allDfWithPreds)
    assert len(allDf) == len(allDf["dst_filename"].unique())
    outputFile = f"{outputDir}/preds_{taskName}"
    allDf.to_csv(outputFile)
    return outputFile


if __name__ == "__main__":
    CrossValPred(1)
