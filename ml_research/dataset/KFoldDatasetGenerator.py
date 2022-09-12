import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision.transforms import Compose
import torchvision
import torch
from tqdm import tqdm
from ml_research.params.PriceRangeParams import PriceRangeParams


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
        if os.path.exists(srcPath):
            pass
        img = Image.open(srcPath)
        transformed = self.transform(img)
        label = targetData["label"].item()
        target = torch.tensor(label, dtype=torch.int64)
        return transformed, target
        print(transformed)


class KFoldDatasetGenerator:
    def __init__(self) -> None:
        srcDfPath = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/KFold_Saloon-4Dr_kfold_10_Front_View.csv"
        self.imgBaseDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vehicle_type/Saloon-4Dr"

        self.srcDf = pd.read_csv(srcDfPath)
        self.trainTransform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (PriceRangeParams.imgMaxSize, PriceRangeParams.imgMaxSize)
                ),
                torchvision.transforms.ToTensor(),
            ]
        )

    def genDataloader(self):
        allFolds = self.srcDf["kfold"].unique()
        allSplit = []
        for foldId in allFolds:
            allDataInFold = self.srcDf[self.srcDf["kfold"] == foldId]
            trainData = allDataInFold[allDataInFold["train_test"] == "train"]
            testData = allDataInFold[allDataInFold["train_test"] == "test"]
            trainDs = KFoldImageDataset(trainData, self.imgBaseDir, self.trainTransform)
            testDs = KFoldImageDataset(testData, self.imgBaseDir, self.trainTransform)
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
