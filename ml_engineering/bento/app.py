from typing import List
import bentoml

from sklearn import svm
from sklearn import datasets
import mlflow
import torch
from ml_research.params import ImportEnv
import torchvision

# modelScript = mlflow.pytorch.load_model(model_uri=f"models:/mrm_OOD_detector/{1}")
# fp = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/mrm_OOD_detector_v1.pth"
# # with open(fp, "rb") as f:
# #     modelScript = torch.jit.load(f, map_location=torch.device("cuda"))
# modelname = f"ood_detector_v1"
# saved_model = bentoml.torchscript.save_model("ood_detector_v1", modelScript)
# print(f"Model saved: {saved_model}")
modelId = "ood_detector_v1:rrkc6tbzj2xnigga"
modelRunner = bentoml.torchscript.load_model(modelId).to_runner()


import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
import ujson as json
from ml_engineering.bento.req import OOD_Detector_Request
from PIL import Image
from io import BytesIO

with open(
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/ml_engineering/bento/index_to_name.json",
    "r",
) as f:
    labelToName = json.load(f)


svc = bentoml.Service("detect", runners=[modelRunner])


@svc.api()
def classify(input_series: OOD_Detector_Request):
    print("start")
    allDataToUpload = []
    for i in input_series.batch_request:
        pilImg = Image.open(BytesIO(i.image_bytes))
        allDataToUpload.append(transform(pilImg).to(device))
    threshold = 0.6
    _, results = modelRunner.predict.run(input_series)
    box_filters = [row["scores"] >= threshold for row in results]
    filtered_boxes, filtered_classes, filtered_scores = [
        [row[key][box_filter].tolist() for row, box_filter in zip(results, box_filters)]
        for key in ["boxes", "labels", "scores"]
    ]

    for classes, boxes, scores in zip(
        filtered_classes, filtered_boxes, filtered_scores
    ):
        retval = []
        for _class, _box, _score in zip(classes, boxes, scores):

            _retval = {
                "confidence": _score,
                "category": labelToName[str(_class)],
                "bbox": _box,
            }

            retval.append(_retval)
    print("Done")
    return retval
