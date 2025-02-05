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
    subfolder_path = os.path.join(root_folder, subfolder)
    if os.path.isdir(subfolder_path):
        sub1.append(subfolder_path)
sub2 = []
root_folder = "C:/eventCamera/Caltech101_annotations"
subfolders = os.listdir(root_folder)
sorted_file_list = sorted(subfolders)
for subfolder in sorted_file_list:
    subfolder_path = os.path.join(root_folder, subfolder)
    if os.path.isdir(subfolder_path):
        sub2.append(subfolder_path)

for i in range(len(sub1)-1):
    folder_path = sub1[i] 
    file_list = os.listdir(folder_path)
    sorted_file_list = sorted(file_list)
    for file_name in sorted_file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  
            file1.append(file_path)

    folder_path = sub2[i]  
    file_list = os.listdir(folder_path)
    sorted_file_list = sorted(file_list)
    for file_name in sorted_file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path): 
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
                 th + datum['x'].item(0)
                if pixel_idx not in polarity_dict:
                    polarity_dict[pixel_idx] = []
                polarity_dict[pixel_idx].append(datum['p'].item(0))

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
                cwtmatr, frequencies = pywt.cwt(sequence, np.arange(1, 4), wavename) 
                len1, len2 = cwtmatr.shape 
                for i1 in range(0, len1, 1):
                    for j1 in range(0, len2, 1):
                        video_real[i1, j1] = cwtmatr[i1, j1].real
                img[n, m] = np.sum(video_real)  
                if img[n, m] < 0: 
                    result_img[n, m] = 255
                else:
                    result_img[n, m] = 0
        cv2.imwrite('/Users/chenyu/Desktop/frame.jpg', result_img)
        image = cv2.imread('/Users/chenyu/Desktop/frame.jpg')  
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0) 
        edged = cv2.Canny(gray, 50, 100)  
        edged = cv2.dilate(edged, None, iterations=1)  
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts) 
        colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))  
        orig = image.copy()
        orig = draw_rect(cnts, orig)  

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

