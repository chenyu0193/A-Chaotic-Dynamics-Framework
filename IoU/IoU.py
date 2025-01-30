import os
import numpy as np
import time
import cv2
import pywt
import math
import imutils
import read_dataset
from imutils import contours
from imutils import perspective
import read_annotation
import draw_rect
import recover_bbox
import calculate_iou

# x_max, y_max, x, y = recover_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
# print(x, y, x_max, y_max)
polarity_dict = {}
p_dict = {}
length = []
area = []
file1 = []
file2 = []
iOU = []
mAP = []
af = 0.1
ae = 1.00
ve = 50
x0 = 0
y0 = 0
z0 = 0
frame_length = 70e3

sub1 = []
root_folder = "C:/eventCamera/Caltech101"
subfolders = os.listdir(root_folder)
sorted_file_list = sorted(subfolders)
for subfolder in sorted_file_list:
    # 使用os.path.join拼接完整的路径
    subfolder_path = os.path.join(root_folder, subfolder)
    # 判断是否为文件夹
    if os.path.isdir(subfolder_path):
        sub1.append(subfolder_path)
sub2 = []
root_folder = "C:/eventCamera/Caltech101_annotations"
subfolders = os.listdir(root_folder)
sorted_file_list = sorted(subfolders)
for subfolder in sorted_file_list:
    # 使用os.path.join拼接完整的路径
    subfolder_path = os.path.join(root_folder, subfolder)
    # 判断是否为文件夹
    if os.path.isdir(subfolder_path):
        sub2.append(subfolder_path)

for i in range(len(sub1)-1):
    folder_path = sub1[i]  # 替换为你的文件夹路径
    # 获取文件夹中的所有文件
    file_list = os.listdir(folder_path)
    # 按文件名进行排序
    sorted_file_list = sorted(file_list)
    # 遍历排序后的文件列表并处理每个文件
    for file_name in sorted_file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # 确保是文件而不是文件夹
            file1.append(file_path)

    folder_path = sub2[i]  # 替换为你的文件夹路径
    # 获取文件夹中的所有文件
    file_list = os.listdir(folder_path)
    sorted_file_list = sorted(file_list)
    # 遍历文件列表并逐个读取文件
    for file_name in sorted_file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # 确保是文件而不是文件夹
            file2.append(file_path)
    print(file1)
    for j in range(len(file2)-1):
        td = read_dataset(file1[j])
        boxs = read_annotation(file2[j])
        x_max, y_max, x_min, y_min = recover_bbox(boxs[0], boxs[1], boxs[2], boxs[3])
        # print(x_min, y_min, x_max, y_max)
        t_max = td.data.ts[-1]
        frame_start = td.data[0].ts
        frame_end = td.data[0].ts + frame_length
        td_img = np.ones((td.height, td.width), dtype=np.uint8)
        frame_data = td.data[(td.data.ts >= frame_start) & (td.data.ts < frame_end)]
        # with timer.Timer() as em_playback_timer:
        for datum in np.nditer(frame_data):
            # td_img[datum['y'].item(0), datum['x'].item(0)] = datum['p'].item(0)
            # print(td_img)
            if 0 <= datum['x'].item(0) < td.width and 0 <= datum['y'].item(0) < td.height:
                # 将事件存储在对应像素的列表中
                pixel_idx = datum['y'].item(0) * td.width + datum['x'].item(0)
                if pixel_idx not in polarity_dict:
                    polarity_dict[pixel_idx] = []
                polarity_dict[pixel_idx].append(datum['p'].item(0))

        # 输出每个像素的极性列表
        for pixel_idx, polarity_list in polarity_dict.items():
            # print(f"Pixel ({pixel_idx % td.width}, {pixel_idx // td.width}): {polarity_list}")
            length.append(len(polarity_list))
        max_length = max(length)

        for pixel_idx, polarity_list in polarity_dict.items():
            # polarity_dict[pixel_idx] = str(polarity_dict[pixel_idx]).ljust(max_length, "0")
            num = polarity_list[len(polarity_list) - 1]
            polarity_list += [num] * (max_length - len(polarity_list))
            # print(f"Pixel ({pixel_idx % td.width}, {pixel_idx // td.width}): {polarity_list}")

        sequence = np.zeros(max_length)
        wavename = "gaus1"
        video_real = np.zeros((3, max_length))
        img = np.zeros((td.height, td.width))
        result_img = np.zeros((td.height, td.width))
        output = np.zeros((td.height, td.width))

        for pixel_idx, polarity_list in polarity_dict.items():
            m = pixel_idx % td.width
            n = pixel_idx // td.width
            if 0 <= m < td.width and 0 <= n < td.height:
                for k in range(max_length):
                    st1 = polarity_list[k]
                    x0 = math.exp(-af) * x0 + st1
                    y0 = math.exp(-ae) * y0 + ve * z0
                    z0 = 1 / (1 + math.exp(-(x0 - y0)))
                    sequence[k] = z0
                cwtmatr, frequencies = pywt.cwt(sequence, np.arange(1, 4), wavename)  # 连续小波变换模块
                len1, len2 = cwtmatr.shape  # 输出为：-0.0532094 +0.11571999j -0.05402121-0.11622215j  0.06683362+0.01142197j...的复数数组，实部有正有负
                for i1 in range(0, len1, 1):
                    for j1 in range(0, len2, 1):
                        video_real[i1, j1] = cwtmatr[i1, j1].real
                img[n, m] = np.sum(video_real)  # 输出为：-0.0532094  -0.05402121  0.06683362 ...,与cwtmatr同shape
                if img[n, m] < 0:  # 输出为混沌信号的实部和大于0，周期信号的实部和小于0
                    result_img[n, m] = 255
                else:
                    result_img[n, m] = 0
        cv2.imwrite('/Users/chenyu/Desktop/frame.jpg', result_img)
        image = cv2.imread('/Users/chenyu/Desktop/frame.jpg')  # 读取图片
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)  # 执行高斯滤波
        edged = cv2.Canny(gray, 50, 100)  # 执行Canny边缘检测
        edged = cv2.dilate(edged, None, iterations=1)  # 执行腐蚀和膨胀处理, 先膨胀后腐蚀，闭运算，填充物体内黑洞，连接邻近物体和平滑边界
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 在边缘映射中寻找轮廓
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)  # 对轮廓点进行排序
        colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))  # 设置显示颜色
        orig = image.copy()
        orig = draw_rect(cnts, orig)  # 画框

        for p, boxs in p_dict.items():
            area.append(p)
        area_max = max(area)
        box = p_dict[area_max]
        a, b = box[0][0]
        c, d = box[0][2]
        # print(a, b, c, d)
        box1 = (x_min, y_min, x_max, y_max)  # (x_min, y_min, x_max, y_max)
        box2 = (a, b, c, d)
        iou_score = calculate_iou(box1, box2)
        # print(iou_score)
        iOU.append(iou_score)
        # print("IoU:", iou_score)
        polarity_dict.clear()
        p_dict.clear()
        length.clear()
        area.clear()
    AP = sum(iOU) / len(iOU)
    mAP.append(AP)
    print(AP)
    file1.clear()
    file2.clear()
    iOU.clear()

mAP = sum(mAP) / len(mAP)
print(f'mAP={mAP:.5f}')

