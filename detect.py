import os
import time
import numpy as np

from PIL import Image
import cv2
from numpy import random
import jittor as jt

from models.yolo import Model
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box
from utils.model_utils import time_synchronized
import models.common as common

jt.flags.use_cuda=1

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def detect(save_img=False):
    imgsz = 416

    # Load model
    model = Model("./configs/yolov3.yaml")
    model.load("./yolov3.pkl")
    model = model.fuse()
    model.eval()

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Get names and colors
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    image_folder_path = "./sample_images"

    image_file_name_list = os.listdir(image_folder_path)

    for image_file_name in image_file_name_list:
        img_source = cv2.imread(os.path.join(image_folder_path, image_file_name))

        img = letterbox(img_source, new_shape=(640, 640))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = jt.array(img,dtype="float32") # uint8 to fp32
        img /= 255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, None, False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = img_source

            detect_result = ""

            gn = jt.array([im0.shape[1],im0.shape[0],im0.shape[1],im0.shape[0]]).float32()  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    detect_result += str(int(n)) + " " + names[int(c.int())] + ", "
                
                # Write results
                for i in reversed(range(len(det))):
                    xyxy = det[i,:4]
                    conf = det[i,4]
                    cls = det[i,5].int()

                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            #  cv2.imshow("test", im0)
            #  cv2.waitKey(0)
            print(detect_result)

if __name__ == '__main__':
    common.Conv.use_v3 = True

    with jt.no_grad():
        detect()
