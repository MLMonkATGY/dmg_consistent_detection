from http.client import HTTP_VERSION_NOT_SUPPORTED
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision.transforms import Compose
import torchvision
import torch
from tqdm import tqdm
from ml_research.eval.OODFilter import loadAndFilter
from ml_research.params.PriceRangeParams import PriceRangeParams
import glob


class KFoldImageDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, imgDir: str, transform: torchvision.transforms
    ):
        self.df = df
        self.imgDir = imgDir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        targetData = self.df.iloc[idx]
        srcPath = os.path.join(self.imgDir, targetData["dst_filename"])
        img = Image.open(srcPath)
        transformed = self.transform(img)
        label = targetData["label"].item()
        target = torch.tensor(label, dtype=torch.int64)
        return transformed, target
        print(transformed)


class KFoldDatasetGenerator:
    def __init__(self, OODCsvPath: str) -> None:
        self.srcDfPath = PriceRangeParams.srcDfPath
        self.imgBaseDir = PriceRangeParams.imgBaseDir
        self.level1FilterDf = pd.read_csv(OODCsvPath)
        oodImg = self.level1FilterDf["rej_filename"].tolist()

        self.srcDf = pd.read_csv(self.srcDfPath)
        beforeSize = len(self.srcDf)

        self.srcDf = self.srcDf[~self.srcDf["dst_filename"].isin(oodImg)]

        if PriceRangeParams.filterRejectImg:
            search = f"{PriceRangeParams.rejLabelDir}/**/*.csv"
            allDfPath = glob.glob(search, recursive=True)
            allDf = []
            for dfPath in allDfPath:
                allDf.append(pd.read_csv(dfPath))
            if len(allDf) > 0:
                rejDf = pd.concat(allDf)
                rejFilename = rejDf["dst_filename"].tolist()
                self.srcDf = self.srcDf[~self.srcDf["dst_filename"].isin(rejFilename)]
                print(f"Rejected not confident : {len(rejFilename)}")
        afterSize = len(self.srcDf)
        removeSize = (beforeSize - afterSize) / 5
        print(f"Rejected {removeSize} images per fold")
        print(f"DS size {afterSize / 5} images per fold")
        print(self.srcDf.groupby("label")["src_path"].count())
        self.trainTransform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (PriceRangeParams.imgMaxSize, PriceRangeParams.imgMaxSize)
                ),
                torchvision.transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                ),
                torchvision.transforms.RandomHorizontalFlip(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.evalTransform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (PriceRangeParams.imgMaxSize, PriceRangeParams.imgMaxSize)
                ),
                torchvision.transforms.ToTensor(),
            ]
        )

    def genDataloader(self):
        allFolds = self.srcDf["kfold"].unique()
        # allFolds = [x for x in allFolds if x > 4]
        allSplit = []
        for foldId in allFolds:
            allDataInFold = self.srcDf[self.srcDf["kfold"] == foldId]
            trainData = allDataInFold[allDataInFold["train_test"] == "train"]
            testData = allDataInFold[allDataInFold["train_test"] == "test"]
            trainDs = KFoldImageDataset(trainData, self.imgBaseDir, self.trainTransform)
            testDs = KFoldImageDataset(testData, self.imgBaseDir, self.evalTransform)
            trainLoader = DataLoader(
                trainDs,
                batch_size=PriceRangeParams.trainBatchSize,
                num_workers=PriceRangeParams.trainCPUWorker,
                shuffle=True,
            )
            testLoader = DataLoader(
                testDs,
                batch_size=PriceRangeParams.trainBatchSize,
                num_workers=PriceRangeParams.trainCPUWorker,
                shuffle=False,
            )
            print(f"Trainset size : {len(trainDs)}")
            print(f"Testset size : {len(testDs)}")

            allSplit.append((trainLoader, testLoader))
        return allSplit


if __name__ == "__main__":
    a = KFoldDatasetGenerator()
    b = a.genDataloader()
    for (trainLoader, testLoader) in b:
        for img, targets in tqdm(trainLoader):
            pass
        for img, targets in tqdm(testLoader):
            pass
        break
    print("All data checking passed")
