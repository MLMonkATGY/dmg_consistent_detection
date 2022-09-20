# from src.ml_engineering.StagingModelInfo import stagingModelInfo
from ml_engineering.DownloadCkpt import DownloadCheckpoints

from ml_engineering.ExportModel import ExportTorchScriptModel
from ml_engineering.StageEval import StageModel

from loguru import logger
import torch


if __name__ == "__main__":
    # downloadedCkptPath = DownloadCheckpoints()
    downloadedCkptPath = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/staging/7977977d391f411c90af6aa89b38408a/epoch=02-map_50=0.29-mar_100=0.62.ckpt"
    logger.success("Ckpt downloaded : {}".format(downloadedCkptPath))
    localTorchScriptPath = ExportTorchScriptModel(downloadedCkptPath)
    logger.success("torch script compiled : {}".format(localTorchScriptPath))

    StageModel(localTorchScriptPath)
    logger.success("model staged sucessfully")
