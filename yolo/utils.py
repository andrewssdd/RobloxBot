# Wrapper functions for yolov5

import torch
import torch.nn as nn
from .yolov5.models.experimental import attempt_load
from .yolov5.utils.torch_utils import select_device
from .yolov5.utils.general import non_max_suppression
from .yolov5.utils.general import scale_coords
from .yolov5.utils.plots import plot_one_box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def loadModel(weights, device):
    ''' load a yolo model
        weight: file name of the weights
        device:
    '''
    model = attempt_load(weights, map_location=device)  # load FP32 model
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    return model

def infer(model, img, device, augment = False, conf_thres = 0.25, iou_thres = 0.45, classes = None, agnostic_nms = False):
    half = device.type != 'cpu'  # half precision only supported on CUDA
    img = np.moveaxis(img, 2, 0)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    return pred, img

def processPrediction(pred, img1, img0, names, colors):
    '''

    :param pred: predictions from yolo model
    :param img1: image (pytorch)
    :param img0: image (orig numpy)
    :param names: labels
    :param colors: colors of label box
    :return: centers of boxes for each label category
    '''
    # Process detections
    pos = {}
    for cls in range(len(names)):
        pos[cls] = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results

            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                pos[int(cls)].append( (float(xyxy[0]+xyxy[2])/2./img0.shape[1], float(xyxy[1]+xyxy[3])/2./img0.shape[0]) )
    return pos

