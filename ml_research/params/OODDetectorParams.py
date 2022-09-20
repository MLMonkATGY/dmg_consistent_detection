from dataclasses import dataclass


@dataclass
class OODDetectorParams:
    imgMinSize: int
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
    imgDir: str
    trainAnnFile: str
    evalAnnFile: str


OODParams = OODDetectorParams(
    imgDir="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/images_neg",
    trainAnnFile="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/train_neg.json",
    evalAnnFile="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/sample/test.json",
    imgMaxSize=700,
    imgMinSize=500,
    trainBatchSize=8,
    trainCPUWorker=5,
    experimentName="mrm_OOD_detecter",
    localSaveDir="mlruns",
    saveTopNBest=5,
    check_val_every_n_epoch=3,
    learningRate=1e-3,
    trainingPrecision=16,
    randomSeed=99,
    maxEpoch=100,
)
