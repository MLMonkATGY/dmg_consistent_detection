from dataclasses import dataclass


@dataclass
class StagingModelInfo:
    runId: str
    taskName: str
    expId: int
    expName: str
    ckptArtifactPath: str
    bucketName: str
    customEvalImgDir: str
    dryRun: bool


stagingModelInfo = StagingModelInfo(
    runId="a3331e5b79f840fa96c7eb6213273321",
    taskName="mrm_price_classifier",
    expId=55,
    ckptArtifactPath="checkpoints/epoch=04-e_acc=0.84-e_0_TP=0.85-e_1_TP=0.80.ckpt",
    bucketName="mlflow",
    expName="mrm_dmg_price_range",
    customEvalImgDir="/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vType_range/SUV-5Dr_RearView_cls",
    dryRun=False,
)
