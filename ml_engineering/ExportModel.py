from dataclasses import dataclass
import dataclasses
from typing import List
import torch
from loguru import logger

# from ml_research.train.OODFilterTrainer import ProcessModel
from ml_research.train.PriceRangeTrainer import ProcessModel

import ujson as json
from ml_engineering.StagingModelInfo import stagingModelInfo
import torchvision


@dataclass
class ModelServingInfo:
    expId: int
    runId: str
    origFileName: str
    batchSize: int
    inputChannel: int
    taskName: str
    expName: str


def ExportTorchScriptModel(ckptPath: str):
    runId = stagingModelInfo.runId

    device = torch.device("cuda")
    localSaveDir = "/".join(ckptPath.split("/")[:-1])
    processModel = ProcessModel.load_from_checkpoint(ckptPath)
    # processModel = ProcessModel()
    processModel.eval()
    batchSize = 1
    inputChannel = 3
    filename = ckptPath.split("/")[-1].replace(".ckpt", "")
    taskName = stagingModelInfo.taskName
    extraInfo = dataclasses.asdict(
        ModelServingInfo(
            expId=str(55),
            runId=runId,
            origFileName=filename,
            batchSize=str(batchSize),
            inputChannel=str(inputChannel),
            taskName=taskName,
            expName=stagingModelInfo.expName,
        )
    )
    serviceModel = processModel.model.half()
    serviceModel = serviceModel.to(device)
    serviceModel.eval()

    # evalLoaderIter = iter(dm.val_dataloader())
    # inputSample, labels = next(evalLoaderIter)
    outputScript = torch.jit.script(
        serviceModel,
    )
    # frozenGrapth = torch.jit.freeze(outputScript)
    filename = ckptPath.split("/")[-1].replace(".ckpt", "_torchscript.pth")
    localTorchScript = "{0}/{1}".format(localSaveDir, filename)
    with open(localTorchScript, "wb") as f:
        torch.jit.save(outputScript, f, extraInfo)
    return localTorchScript
