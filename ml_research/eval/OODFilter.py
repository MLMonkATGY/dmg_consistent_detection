import cv2
import numpy as np
import torch
import glob
import time
import os
from ml_research.params.PriceRangeParams import PriceRangeParams
from ml_research.train.TrainModelProcess import create_model
from ml_research.eval.visualize import visualizeAndFilter


def loadAndFilter():
    PATH = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/alternative_detection/control_angle_detection_e40_0.56.ckpt"
    outputDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/inference_outputs/images"

    model = create_model(5)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    rejCsvPath = visualizeAndFilter(model, 20, PriceRangeParams.imgBaseDir, outputDir)
    return rejCsvPath


if __name__ == "__main__":
    loadAndFilter()
