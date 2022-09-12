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
    experimentName: str
    localSaveDir: str
    saveTopNBest: int
    check_val_every_n_epoch: int
    learningRate: float
    trainingPrecision: int
    randomSeed: int
    maxEpoch: int


PriceRangeParams = Params(
    trainAnnFilePath="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/complete.json",
    trainDataFolder="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/images",
    evalAnnFilePath="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/complete.json",
    evalDataFolder="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/images",
    imgMaxSize=480,
    trainBatchSize=24,
    trainCPUWorker=6,
    experimentName="mrm_dmg_price_range",
    localSaveDir="mlruns",
    saveTopNBest=5,
    check_val_every_n_epoch=1,
    learningRate=1e-3,
    trainingPrecision=16,
    randomSeed=99,
    maxEpoch=20,
)
