# avodown
# Copyright (c) 2025 gzqccnu 
#
# This program is released under the terms of the Apache License.
# See https://opensource.org/licenses/Apache for more information.
#
# Project homepage: https://github.com/gzqccnu/avodown
# Description: Using models to avoid the old fall down

import os
import cv2
import time
import random
import torch
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.plots import plot_one_box

# 导入 OpenPose 以及动作分类相关函数
import openpose
from keypoints import extract_keypoints, group_keypoints
from pose import Pose
from act import act
from utils.contrastImg import coincide
from math import ceil, floor

class FallDetectionModel:
    def __init__(self):
        self.yolo_model = None
        self.openpose_model = None
        self.action_model = None
        self.device = None
        self.half = False
        self.loaded_weights = None
        self.use_npu = False
        self.img_size = None

    def load(self, weights='models/yolo.pt', device='cpu', use_npu=False, img_size=640):
        if self.yolo_model is not None and self.loaded_weights == weights:
            print("Using cached YOLO model.")
            return

        self.img_size = img_size
        if use_npu:
            os.environ['NPU_VISIBLE_DEVICES'] = '1'
            self.device = torch.device('npu:1')
            torch.npu.set_device(self.device)
        else:
            self.device = select_device(device)
        print(f"Loading YOLO on device: {self.device}")
        self.yolo_model = attempt_load(weights, map_location=self.device)
        self.yolo_model.to(self.device)
        self.half = self.device.type != 'cpu' and not use_npu
        if self.half:
            self.yolo_model.half()

        # 加载 OpenPose 和动作分类模型
        map_loc = self.device if use_npu else torch.device('cpu')
        self.openpose_model = torch.load('./openpose.jit', map_location=map_loc)
        self.action_model = torch.load('./action.jit', map_location=map_loc)

        self.loaded_weights = weights
        self.use_npu = use_npu
        # 预热 YOLO
        torch.npu.set_compile_mode(jit_compile=True)
        dummy = torch.zeros((1, 3, img_size, img_size), device=self.device)
        with torch.no_grad():
            _ = self.yolo_model(dummy.half() if self.half else dummy)
        print("Models loaded and warmed up.")
        torch.npu.set_compile_mode(jit_compile=False)

    def predict_frame(self, frame: np.ndarray,
                      conf_thres=0.5,
                      iou_thres=0.45,
                      augment=False,
                      classes=[0],
                      agnostic_nms=False):
        """
        单帧YOLO检测，返回 [{'label','confidence','box'}]
        """
        img, _, _ = letterbox(frame, new_shape=self.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            pred = self.yolo_model(tensor, augment=augment)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)

        names = self.yolo_model.module.names if hasattr(self.yolo_model, 'module') else self.yolo_model.names
        results = []
        for det in pred:
            if det is None or len(det) == 0:
                continue
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                results.append({
                    'label': names[int(cls)],
                    'confidence': float(conf),
                    'box': [x1, y1, x2, y2]
                })
        return results

    # def predict_with_fall(self, frame: np.ndarray,
    #                       conf_thres=0.5,
    #                       iou_thres=0.45,
    #                       augment=False,
    #                       classes=[0],
    #                       agnostic_nms=False) -> dict:
    #     """
    #     执行 YOLO + OpenPose + 动作分类，返回检测结果与摔倒标志
    #     """
    #     # 1. YOLO 检测
    #     dets = self.predict_frame(frame, conf_thres, iou_thres, augment, classes, agnostic_nms)

    #     fall_flag = False
    #     # OpenPose 参数
    #     height_size = 256
    #     stride = 8
    #     upsample_ratio = 4
    #     num_kpts = Pose.num_kpts

    #     for det in dets:
    #         if det['label'] != 'person':
    #             continue
    #         x1, y1, x2, y2 = det['box']
    #         crop = frame[y1:y2, x1:x2]
    #         if crop.size == 0:
    #             continue

    #         # 2. 提取关键点 heatmaps & pafs
    #         heatmaps, pafs, scale, pad = openpose.infer_fast(
    #             self.openpose_model, crop,
    #             net_input_height_size=height_size,
    #             stride=stride,
    #             upsample_ratio=upsample_ratio,
    #             cpu=False,
    #             device=self.device
    #         )
    #         # 3. 解析关键点并映射回原图
    #         all_kpts = []
    #         total = 0
    #         for k in range(num_kpts):
    #             total += extract_keypoints(heatmaps[:, :, k], all_kpts, total)
    #         pose_entries, kpts = group_keypoints(all_kpts, pafs, demo=False)

    #         for entry in pose_entries:
    #             coords = np.zeros((num_kpts, 2), dtype=int)
    #             for idx in range(num_kpts):
    #                 kp_id = entry[idx]
    #                 if kp_id != -1:
    #                     coords[idx] = kpts[int(kp_id)][:2].astype(int)
    #             pose = Pose(coords, entry[18])
    #             # 4. 绘制姿态图给 act 使用
    #             pose.img_pose = pose.draw(crop.copy(), is_save=False, show_draw=False)
    #             # 5. 动作识别
    #             pose = act(self.action_model, pose, pose.bbox[2]/pose.bbox[3], device=self.device)
    #             if pose.pose_action == 'fall':
    #                 fall_flag = True
    #                 break
    #         if fall_flag:
    #             break

    #     return {
    #         'detections': dets,
    #         'fall_detected': fall_flag
    #     }

    def predict_with_fall(self,
                      frame: np.ndarray,
                      conf_thres: float = 0.5,
                      iou_thres: float = 0.45,
                      augment: bool = False,
                      classes: list[int] = [0],
                      agnostic_nms: bool = False) -> dict:
        """
        对单帧 BGR ndarray 执行 YOLO→OpenPose(run_demo)→返回
        {
            'detections': [ {'label','confidence','box'}, … ],
            'fall_detected': True/False
        }
        """
        # 1. YOLO 检测
        dets = self.predict_frame(
            frame,
            conf_thres  = conf_thres,
            iou_thres   = iou_thres,
            augment     = augment,
            classes     = classes,
            agnostic_nms= agnostic_nms
        )

        detections = []
        boxList    = []

        # 2. 收集所有检测、同时对 person 做长宽比初筛
        for d in dets:
            x1,y1,x2,y2 = d['box']
            detections.append({
                'label':      d['label'],
                'confidence': d['confidence'],
                'box':        [x1, y1, x2, y2]
            })
            if d['label'] == 'person':
                w, h = x2 - x1, y2 - y1
                if h > 0 and (w / h) >= 0.8:
                    boxList.append([x1, y1, x2, y2])

        # 3. 如果有候选框，就调用 run_demo 进行最精确的姿态+act摔倒预测
        fall_flag = False
        if boxList:
            fall_flag = openpose.run_demo(
                self.openpose_model,
                self.action_model,
                [frame],       # run_demo expects a list of images
                256,           # height_size
                True,         # cpu flag
                boxList,
                device=self.device
            )

        return {
            'detections':     detections,
            'fall_detected':  fall_flag
        }
