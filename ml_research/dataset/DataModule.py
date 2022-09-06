import pytorch_lightning as pl
from typing import Dict, Optional, Any, List, Tuple
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from ml_research.params.hyperparams import GlobalParams
import albumentations as A
import cv2

from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader2
import torch
import numpy as np
from ml_research.dataset.CocoDataset import CocoDataset


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def collaterFn(input: Dict):
    allImgTensor = torch.stack([x["transformed"]["image"] for x in input])
    max_num_annots = max(len(x["transformed"]["category_ids"]) for x in input)

    annot_padded = torch.ones((len(input), max_num_annots, 5)) * -1

    for sampleId, x in enumerate(input):
        for bboxId, (rawLabel, rawBbox) in enumerate(
            zip(x["transformed"]["category_ids"], x["transformed"]["bboxes"])
        ):
            bboxLabelList = list(rawBbox)
            bboxLabelList[2] = bboxLabelList[0] + bboxLabelList[2]
            bboxLabelList[3] = bboxLabelList[1] + bboxLabelList[3]

            bboxLabelList.append(rawLabel)
            ann = np.array(bboxLabelList)
            sampleAnnTensor = torch.from_numpy(ann)
            annot_padded[sampleId, bboxId, :] = sampleAnnTensor

    return allImgTensor, annot_padded


class CocoDatamodule(pl.LightningDataModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.trainAnn = GlobalParams.trainAnnFilePath
        self.trainDataDir = GlobalParams.trainDataFolder
        self.evalAnn = GlobalParams.evalAnnFilePath
        self.evalDataFolder = GlobalParams.evalDataFolder
        self.imgMaxSize = GlobalParams.imgMaxSize
        self.trainTransform = A.Compose(
            [
                # A.PadIfNeeded(
                #     min_height=hyperparams.imgMaxSize,
                #     min_width=hyperparams.imgMaxSize,
                #     border_mode=cv2.BORDER_CONSTANT,
                #     value=255,
                # ),
                A.Resize(
                    GlobalParams.imgMaxSize,
                    GlobalParams.imgMaxSize,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.Normalize(),
                A.Affine(
                    scale=(0.8, 0.99), translate_percent=0.1, rotate=(0, 5), p=0.2
                ),
                A.GaussianBlur(blur_limit=(5, 5), p=0.2),
                A.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.2
                ),
                A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.2),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                min_visibility=0.1,
                label_fields=["category_ids"],
            ),
        )
        self.valTransform = A.Compose(
            [
                # A.PadIfNeeded(
                #     min_height=hyperparams.imgMaxSize,
                #     min_width=hyperparams.imgMaxSize,
                #     border_mode=cv2.BORDER_CONSTANT,
                #     value=255,
                # ),
                A.Resize(
                    GlobalParams.imgMaxSize,
                    GlobalParams.imgMaxSize,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                min_visibility=0.1,
                label_fields=["category_ids"],
            ),
        )
        self.batch_size = GlobalParams.trainBatchSize
        self.workerNum = GlobalParams.trainCPUWorker
        self.trainSet = CocoDataset(
            self.trainAnn, self.trainDataDir, self.trainTransform
        )
        self.evalSet = CocoDataset(self.evalAnn, self.evalDataFolder, self.valTransform)

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return DataLoader2(
            self.trainSet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workerNum,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader2(
            self.evalSet,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        ...


if __name__ == "__main__":
    dm = CocoDatamodule()
    trainLoader = dm.train_dataloader()
    for i in trainLoader:
        print(i)
