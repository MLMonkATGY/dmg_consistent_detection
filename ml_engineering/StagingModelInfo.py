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
    runId="817c94f314384ac4873dc002421541e1",
    taskName="mrm_OOD_detector",
    expId=56,
    ckptArtifactPath="checkpoints/epoch=68-map_50=0.70-mar_100=0.70.ckpt",
    bucketName="mlflow",
    expName="mrm_OOD_detecter",
)
