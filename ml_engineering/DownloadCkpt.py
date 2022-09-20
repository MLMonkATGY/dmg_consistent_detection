import mlflow
import os
from mlflow.tracking import MlflowClient
import urllib3
from urllib3.response import HTTPResponse
from minio import Minio
from ml_engineering.StagingModelInfo import stagingModelInfo

def getMinioConnection():
    minioHost = "192.168.1.3:9000"
    client = Minio(
        minioHost, access_key="alextay96", secret_key="Iamalextay96", secure=False
    )
    return client


def DownloadCheckpoints():
    baseOutputDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/staging"
    expId = stagingModelInfo.expId
    runId = stagingModelInfo.runId
    artifactPath = stagingModelInfo.ckptArtifactPath
    s3Client = getMinioConnection()
    bucketName = stagingModelInfo.bucketName
    s3ObjName = "{0}/{1}/artifacts/{2}".format(expId, runId, artifactPath)
    localModelName = artifactPath.split("/")[-1]
    localSavePath = os.path.join(baseOutputDir, runId, localModelName)
    s3Client.fget_object(bucketName, s3ObjName, localSavePath)
    return localSavePath
    