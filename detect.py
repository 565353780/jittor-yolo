import time
from pathlib import Path

import cv2
from numpy import random
import jittor as jt

from models.yolo import Model
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box
from utils.model_utils import time_synchronized
import models.common as common

jt.flags.use_cuda=1

def detect(save_img=False):
    imgsz = 640

    # Load model
    model = Model("./configs/yolov3.yaml")
    model.load("./yolov3.pkl")
    model = model.fuse()
    model.eval()

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Set Dataloader
    dataset = LoadImages("./sample_images", img_size=imgsz)

    # Get names and colors
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    image_folder_path = "./sample_images"

    image_file_name_list = os.listdir(image_folder_path)

    for image_file_name in image_file_name_list:
        img = cv2.imread(os.path.join(image_folder_path, image_file_name))
        img /= 255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, None, False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s

            gn = jt.array([im0.shape[1],im0.shape[0],im0.shape[1],im0.shape[0]]).float32()  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                
                # Write results
                for i in reversed(range(len(det))):
                    xyxy = det[i,:4]
                    conf = det[i,4]
                    cls = det[i,5].int()
                    #  if save_txt:  # Write to file
                    #      xywh = (xyxy2xywh(jt.array(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #      line = (cls, *xywh, conf) # label format
                    #      with open(txt_path + '.txt', 'a') as f:
                    #          f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #  if save_img or view_img:  # Add bbox to image
                    #      label = f'{names[int(cls)]} {conf:.2f}'
                    #      plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            #  cv2.imshow(str(p), im0)


    # Run inference
    for path, img, im0s, vid_cap in dataset:
        img = jt.array(img,dtype="float32") # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndim == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, None, False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s

            gn = jt.array([im0.shape[1],im0.shape[0],im0.shape[1],im0.shape[0]]).float32()  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                
                # Write results
                for i in reversed(range(len(det))):
                    xyxy = det[i,:4]
                    conf = det[i,4]
                    cls = det[i,5].int()
                    #  if save_txt:  # Write to file
                    #      xywh = (xyxy2xywh(jt.array(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #      line = (cls, *xywh, conf) # label format
                    #      with open(txt_path + '.txt', 'a') as f:
                    #          f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #  if save_img or view_img:  # Add bbox to image
                    #      label = f'{names[int(cls)]} {conf:.2f}'
                    #      plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            #  cv2.imshow(str(p), im0)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    common.Conv.use_v3 = True

    with jt.no_grad():
        detect()
