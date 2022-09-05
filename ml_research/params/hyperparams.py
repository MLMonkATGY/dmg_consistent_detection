from dataclasses import dataclass


@dataclass
class Params:
    trainAnnFilePath: str
    trainDataFolder: str
    evalAnnFilePath: str
    evalDataFolder: str
    imgMaxSize: int
    trainBatchSize: int
    trainCPUWorker: int


GlobalParams = Params(
    trainAnnFilePath="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/complete.json",
    trainDataFolder="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/images",
    evalAnnFilePath="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/complete.json",
    evalDataFolder="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/images",
    imgMaxSize=512,
    trainBatchSize=10,
    trainCPUWorker=6,
)
