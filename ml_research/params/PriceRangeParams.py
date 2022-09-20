from dataclasses import dataclass


@dataclass
class Params:
    srcDfPath: str
    imgBaseDir: str
    rejLabelDir: str
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
    filterRejectImg: bool
    filterIteration: int


PriceRangeParams = Params(
    srcDfPath="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/kfold_src/KFold_MPV_RearView_cls_kfold_5_RearView.csv",
    imgBaseDir="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vType_range/MPV_RearView_cls",
    rejLabelDir="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/wrong_label",
    imgMaxSize=350,
    trainBatchSize=70,
    trainCPUWorker=5,
    experimentName="mrm_dmg_price_range",
    localSaveDir="mlruns",
    saveTopNBest=3,
    check_val_every_n_epoch=1,
    learningRate=1e-3,
    trainingPrecision=16,
    randomSeed=99,
    maxEpoch=20,
    filterRejectImg=True,
    filterIteration=3,
)
