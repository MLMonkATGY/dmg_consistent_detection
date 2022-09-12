import os
import mlflow
import torch

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.1.3:9000"
os.environ["MLFLOW_TRACKING_USERNAME"] = "alextay96"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "Ilovecoffee96@"
os.environ["MLFLOW_TRACKING_URI"] = "http://192.168.1.3:5000/"

os.environ["AWS_ACCESS_KEY_ID"] = "alextay96"
os.environ["AWS_SECRET_ACCESS_KEY"] = "Iamalextay96"

mlflow.set_tracking_uri("http://192.168.1.3:5000/")
