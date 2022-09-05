import os
from typing import Any, Callable, Dict
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import albumentations as A
from dataclasses import dataclass


class CocoTrainDataset(Dataset):
    def __init__(self, annFilePath: str, dataFolder: str, transform: A.Compose):

        self.transform: A.Compose = transform
        self.annFilePath = annFilePath
        self.dataFolder = dataFolder
        self.coco = COCO(annotation_file=self.annFilePath)
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.classes = {}
        for c in categories:
            self.classes[c["name"]] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {"img": img, "annot": annot}
        bbox = annot[:, :4]
        category_ids = annot[:, -1]

        transformed = self.transform(image=img, bboxes=bbox, category_ids=category_ids)
        sample = {"transformed": transformed, "orig_annot": annot}
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.dataFolder, image_info["file_name"])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False
        )
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            # if a["bbox"][2] < 1 or a["bbox"][3] < 1:
            #     continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a["bbox"]
            annotation[0, 4] = a["category_id"] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        # annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        # annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations
