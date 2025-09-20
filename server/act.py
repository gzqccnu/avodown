# avodown
# Copyright (c) 2025 gzqccnu 
#
# This program is released under the terms of the Apache License.
# See https://opensource.org/licenses/Apache for more information.
#
# Project homepage: https://github.com/gzqccnu/avodown
# Description: Using models to avoid the old fall down

import numpy as np
import torch # 确保导入 torch
import torch_npu # 如果直接操作NPU，也需要导入
from torch import from_numpy, argmax
import os

# DEVICE = "cpu" # 这行可以删除，或者作为默认值，但实际设备通过参数传递

os.environ['NPU_VISIBLE_DEVICES'] = '1'

def act(net, pose, crown_proportion, device=None): # 添加 device 参数
    # img = cv2.cvtColor(pose.img_pose,cv2.IMREAD_GRAYSCALE) # 如果 pose.img_pose 已经是灰度图或你不需要转换，可以注释掉

    maxHeight = pose.keypoints.max()
    minHeight = pose.keypoints.min()

    # 确保 pose.img_pose 是一个可以转换为 NumPy 数组的类型
    # 如果它已经是 NumPy 数组，则不需要额外处理
    # 如果它是 PyTorch Tensor，需要先转为 NumPy 再处理，或者直接操作 Tensor
    # 假设 pose.img_pose 是一个 NumPy 数组
    img_np = pose.img_pose.reshape(-1) # reshape 到一维，如果需要
    img_np = img_np / 255.0 # 数据归一化到 [0, 1]
    img_np = np.float32(img_np)

    # *** 关键修改：将张量移动到正确的设备 ***
    # 使用 from_numpy 创建张量，并直接将其移动到指定的设备
    img_tensor = from_numpy(img_np[None, :]) # None 用于添加 batch 维度

    if device is not None:
        img_tensor = img_tensor.to(device)
    else: # Fallback to CPU if device is not explicitly given (though it should be)
        img_tensor = img_tensor.cpu()
        print("警告: act 函数未收到设备参数，默认使用 CPU。") # 调试信息

    predect = net(img_tensor) # 使用移动到正确设备的张量进行推理

    action_id = int(argmax(predect, dim=1).cpu().detach().item()) # 结果可以移回CPU进行处理

    possible_rate = 0.6 * predect[:, action_id] + 0.4 * (crown_proportion - 1)

    possible_rate = possible_rate.cpu().detach().numpy()[0] # 结果移回CPU进行numpy转换

    if possible_rate > 0.55:
        pose.pose_action = 'fall'
        if possible_rate > 1:
            possible_rate = 1
        pose.action_fall = possible_rate
        pose.action_normal = 1 - possible_rate
    else:
        pose.pose_action = 'normal'
        if possible_rate >= 0.5:
            pose.action_fall = 1 - possible_rate
            pose.action_normal = possible_rate
        else:
            pose.action_fall = possible_rate
            pose.action_normal = 1 - possible_rate

    return pose
