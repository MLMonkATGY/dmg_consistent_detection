from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ml_research.train.PriceRangeTrainer import ProcessModel
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision
import glob
import pandas as pd
import os
from PIL import Image
import torch
import cv2
import numpy as np
from tqdm import tqdm
import shutil

outDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/interpret"
if os.path.exists(outDir):
    shutil.rmtree(outDir)
os.makedirs(outDir, exist_ok=True)

ckpt = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/epoch=013-e_acc=0.84-e_0_TP=0.91-e_1_TP=0.71.ckpt"
level1Rej = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/level1_reject/Pickup-4DrDbl_RearView_cls/rej_Pickup-4DrDbl_RearView_cls.csv"
rejDf = pd.read_csv(level1Rej)
allRejFilename = rejDf["rej_filename"].tolist()
# allRejFilename = []
notConfidentPath = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/wrong_label/Pickup-4DrDbl_RearView_cls"
notConfidentCsv = glob.glob(f"{notConfidentPath}/**/*.csv", recursive=True)
allDf = []
for i in notConfidentCsv:
    allDf.append(pd.read_csv(i))
notConfDf = pd.concat(allDf)
notConfidence = notConfDf["dst_filename"].tolist()
# notConfidence = []
device = torch.device("cuda")
model = ProcessModel.load_from_checkpoint(ckpt)
model = model.eval()
model = model.to(device)
targetLayers = model.model.features[-1]
cam = GradCAMPlusPlus(model=model, target_layers=targetLayers, use_cuda=True)
df = pd.read_csv(
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/KFold_Pickup-4DrDbl_RearView_cls_kfold_5_RearView.csv"
)
targetLabel = 0
df = df[
    (df["kfold"] == 3) & (df["train_test"] == "test") & (df["label"] == targetLabel)
]
srcImgDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/vType_range/Pickup-4DrDbl_RearView_cls"

allImgPath = df["dst_filename"].tolist()
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((350, 350)),
        torchvision.transforms.ToTensor(),
    ]
)
correct = 0
totalImg = 0
for filename in tqdm(allImgPath):
    if filename in allRejFilename or filename in notConfidence:
        continue
    imgSrc = os.path.join(srcImgDir, filename)
    label = df[df["dst_filename"] == filename]["label"].item()
    image = Image.open(imgSrc)
    image2 = cv2.imread(imgSrc)
    orig_image = image2.copy()
    # BGR to RGB
    image2 = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image2 /= 255.0
    # bring color channels to front
    # image2 = np.transpose(image2, (2, 0, 1)).astype(np.float32)

    transformed = transform(image)
    transformed = transformed.to(device)

    image_float_np = transformed.permute(1, 2, 0).cpu().numpy()

    transformed = transformed.unsqueeze(0)
    logit = model(transformed)
    predLabel = torch.argmax(logit)
    targets = [ClassifierOutputTarget(predLabel)]
    totalImg += 1
    if predLabel.item() == label:
        correct += 1
    predLabelNp = predLabel.cpu().numpy().item()
    grayscale_cam = cam(input_tensor=transformed, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=False)
    cv2.imwrite(f"{outDir}/p={predLabelNp}_{filename}", visualization)

print(f"GT label : {targetLabel}")
print(f"Total raw img : {len(allImgPath)}")
print(f"totalImg : {totalImg}")

print(f"Correct : {correct}")

print(f"Acc : {correct / (totalImg)}")
