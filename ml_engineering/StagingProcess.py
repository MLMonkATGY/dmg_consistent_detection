# from src.ml_engineering.StagingModelInfo import stagingModelInfo
from ml_engineering.DownloadCkpt import DownloadCheckpoints

from ml_engineering.ExportModel import ExportTorchScriptModel
from ml_engineering.StageEval import StageModel

from loguru import logger
import torch

from ml_engineering.StagePriceClassifer import StagePriceRangeClassifier


if __name__ == "__main__":
    downloadedCkptPath = DownloadCheckpoints()
    logger.success("Ckpt downloaded : {}".format(downloadedCkptPath))
    localTorchScriptPath = ExportTorchScriptModel(downloadedCkptPath)
    logger.success("torch script compiled : {}".format(localTorchScriptPath))
    StagePriceRangeClassifier(localTorchScriptPath)
    logger.success("model staged sucessfully")
