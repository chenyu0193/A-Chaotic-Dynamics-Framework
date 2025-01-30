import os
import cv2
import pywt
import math
import struct
import asyncio
import numpy as np
import getDVSeventsDavis

sub1 = []
save_dir = "C:/Object Detection/ASL-Frames"
label_list = os.listdir("C:/Object Detection/ASL-DVS")
for j in range(len(label_list)):
    label = label_list[j]
    label_dir = os.path.join(save_dir, str(label))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if os.path.isdir(label_dir):
        sub1.append(label_dir)

sub2 = []
root_folder = "C:/Object Detection/ASL-DVS"
subfolders = os.listdir(root_folder)
# sorted_file_list = sorted(subfolders)
for subfolder in subfolders:
    # 使用os.path.join拼接完整的路径
    subfolder_path = os.path.join(root_folder, subfolder)
    # 判断是否为文件夹
    if os.path.isdir(subfolder_path):
        sub2.append(subfolder_path)

async def my_function():
    af, ae, ve = 0.1, 1.00, 50
    exp_af, exp_ae = math.exp(-af), math.exp(-ae)
    file1 = []
    polarity_dict = {}
    max_length = 30
    wavename = "gaus1"

    for i, folder_path in enumerate(sub2[4:], start=4):
        # 获取文件列表并排序
        file_list = sorted(os.listdir(folder_path))
        file1 = [os.path.join(folder_path, f) for f in file_list if os.path.isfile(os.path.join(folder_path, f))]

        label_dir = sub1[i]
        for j, file_path in enumerate(file1[13:], start=13):
            idx = file_list[j]
            print(f"Processing file: {file_path}")

            T, X, Y, Pol = getDVSeventsDavis(file_path)  # 读取事件数据
            T, X, Y, Pol = map(np.array, [T, X, Y, Pol])
            T, X, Y, Pol = T.flatten(), X.flatten(), Y.flatten(), Pol.flatten()

            step_time = (T[-1] - T[0]) // 840  # 计算每帧时间跨度
            start_idx, start_time, end_time = 0, T[0], T[0] + step_time
            img_count = 0

            while end_time <= T[-1]:
                # 提取当前帧内的事件
                end_idx = np.searchsorted(T, end_time)
                x, y, p = X[start_idx:end_idx], Y[start_idx:end_idx], Pol[start_idx:end_idx]

                # 初始化图像
                width, height = int(X.max() + 1), int(Y.max() + 1)
                img, result_img = np.zeros((height, width)), np.zeros((height, width))

                # 事件极性填充到像素字典
                pixel_indices = y * width + x
                for p_idx, polarity in zip(pixel_indices, p):
                    if p_idx not in polarity_dict:
                        polarity_dict[p_idx] = []
                    polarity_dict[p_idx].append(polarity)

                # 填充和处理极性数据
                for pixel_idx, polarity_list in polarity_dict.items():
                    polarity_list += [polarity_list[-1]] * (max_length - len(polarity_list))
                    polarity_list = polarity_list[:max_length]  # 确保列表长度不超过 max_length

                    # 连续小波变换
                    sequence = np.zeros(max_length)
                    x0, y0, z0 = 0, 0, 0
                    for k, st1 in enumerate(polarity_list):
                        x0 = exp_af * x0 + st1
                        y0 = exp_ae * y0 + ve * z0
                        z0 = 1 / (1 + math.exp(-(x0 - y0)))
                        sequence[k] = z0

                    cwtmatr, _ = pywt.cwt(sequence, np.arange(1, 4), wavename)
                    m, n = int(pixel_idx % width), int(pixel_idx // width)
                    img[n, m] = np.sum(cwtmatr.real)
                    result_img[n, m] = 255 if img[n, m] < 0 else 0

                # 保存结果图像
                result_img = np.flip(result_img, 0)
                img_count += 1
                frame_path = os.path.join(label_dir, f"{idx}_{img_count}.png")
                print(f"Saving frame: {frame_path}")
                cv2.imwrite(frame_path, result_img)

                # 清理字典并更新索引
                polarity_dict.clear()
                start_time = end_time
                end_time += step_time
                start_idx = end_idx

        file1.clear()

# 运行异步函数
asyncio.run(my_function())
