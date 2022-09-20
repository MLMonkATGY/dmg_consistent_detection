from dataclasses import dataclass


@dataclass
class StagingModelInfo:
    runId: str
    taskName: str
    expId: int
    expName: str
    ckptArtifactPath: str
    bucketName: str


stagingModelInfo = StagingModelInfo(
    runId="7977977d391f411c90af6aa89b38408a",
    taskName="mrm_OOD_detector",
    expId=56,
    ckptArtifactPath="checkpoints/epoch=02-map_50=0.29-mar_100=0.62.ckpt",
    bucketName="mlflow",
    expName="mrm_OOD_detecter",
)
