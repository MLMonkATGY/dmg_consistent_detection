import streamlit as st
import bentoml
from PIL import Image
import torchvision
from io import BytesIO
import torch

if __name__ == "__main__":
    allDataToUpload = []
    modelId = "ood_detector_v1:rrkc6tbzj2xnigga"
    modelRunner = bentoml.torchscript.get(modelId).to_runner()
    svc = bentoml.Service("detect", runners=[modelRunner])
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    modelRunner.init_local()
    uploaded_files = st.file_uploader("Choose a Img file", accept_multiple_files=True)
    device = torch.device("cuda")
    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            pilImg = Image.open(BytesIO(bytes_data))
            allDataToUpload.append(transform(pilImg).to(device))
        _, results = modelRunner.run(allDataToUpload)
        print(results)
