from logging import shutdown
import shutil
import cv2
import numpy as np
import torch
import glob
import time
import os
from tqdm import tqdm
import pandas as pd


def visualizeAndFilter(model, epoch: int, DIR_TEST: str, outputDir: str):
    model.eval()

    DEVICE = torch.device("cuda")
    model = model.to(DEVICE)
    CLASSES = [
        "bg",
        "front_side_view",
        "front_view",
        "rear_side_view",
        "rear_view",
    ]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    clsName = DIR_TEST.split("/")[-1]
    rejBaseDir = f"/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/OOD_reject/{clsName}"
    rejImgDir = f"{rejBaseDir}/images"

    os.makedirs(rejImgDir, exist_ok=True)
    os.makedirs(outputDir, exist_ok=True)

    test_images = glob.glob(f"{DIR_TEST}/*.JPG")
    print(f"Test instances: {len(test_images)}")

    detection_threshold = 0.6
    frame_count = 0
    total_fps = 0
    notConfidentImg = []

    for i in tqdm(range(len(test_images))):
        # get the image file name for saving output later on
        image_name = test_images[i].split(os.path.sep)[-1].split(".")[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        end_time = time.time()
        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1
        # load all detection to CPU for further operations
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]["boxes"]) != 0:
            boxes = outputs[0]["boxes"].data.numpy()
            scores = outputs[0]["scores"].data.numpy()
            # filter out boxes according to `detection_threshold`

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            confidence_score = scores[scores >= detection_threshold]
            # if ii % 20 == 0:
            #     print(f"{scores} {boxes}")
            # ii += 1
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]["labels"].cpu().numpy()]
            if len(boxes) == 0:
                # print("rej detected")
                notConfidentImg.append(test_images[i])
                continue
            # draw the bounding boxes and write the class name on top of it
            for j, (box, confidence) in enumerate(zip(draw_boxes, confidence_score)):
                class_name = f"{pred_classes[j]} {np.format_float_positional(confidence, precision=2)}"
                color = COLORS[CLASSES.index(pred_classes[j])]
                cv2.rectangle(
                    orig_image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    2,
                )
                cv2.putText(
                    orig_image,
                    class_name,
                    (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    1,
                    lineType=cv2.LINE_AA,
                )
            # cv2.imshow("Prediction", orig_image)
            # cv2.waitKey(1)
            cv2.imwrite(
                f"{outputDir}/{image_name}_e{epoch}.jpg",
                orig_image,
            )
        else:
            notConfidentImg.append(test_images[i])

        # print(f"Image {i+1} done...")
        # print("-" * 50)
    print("TEST PREDICTIONS COMPLETE")
    # cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    print(f"rej detected : {len(notConfidentImg)}")
    allRejFilename = []
    for rejImg in notConfidentImg:
        filename = rejImg.split("/")[-1]
        allRejFilename.append(filename)
        shutil.copy(rejImg, rejImgDir)
    df = pd.DataFrame(allRejFilename, columns=["rej_filename"])
    rejDfPath = f"{rejBaseDir}/rej_{clsName}.csv"
    df.to_csv(rejDfPath)
    return rejDfPath
