import os
import cv2
import pywt
import math
import asyncio
import numpy as np
import read_dataset

polarity_dict = {}
p_dict = {}
length = []
area = []
file1 = []
file2 = []
file3 = []
iOU = []
mAP = []
af = 0.1
ae = 1.00
ve = 50
x0 = 0
y0 = 0
z0 = 0
frame_length = 10e4
exp_af = math.exp(-af)
exp_ae = math.exp(-ae)

sub1 = []
save_dir = "C:/Object Detection/N-MNIST-Frames/MNIST-F"
label_list = os.listdir("C:/Object Detection/N-MNIST/Train")
# save_dir = "C:/Object Detection/N-caltech-101"
# label_list = os.listdir("C:/eventCamera/Caltech101")
for j in range(len(label_list)):
    label = label_list[j]
    label_dir = os.path.join(save_dir, str(label))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if os.path.isdir(label_dir):
        sub1.append(label_dir)

sub2 = []
# root_folder = "C:/eventCamera/Caltech101"
root_folder = "C:/Object Detection/N-MNIST/N-MNIST"
subfolders = os.listdir(root_folder)
# sorted_file_list = sorted(subfolders)
for subfolder in subfolders:
    subfolder_path = os.path.join(root_folder, subfolder)
    if os.path.isdir(subfolder_path):
        sub2.append(subfolder_path)
async def my_function():
    polarity_dict = {}
    p_dict = {}
    length = []
    area = []
    file1 = []
    af = 0.1
    ae = 1.00
    ve = 50
    x0 = 0
    y0 = 0
    z0 = 0
    exp_af = math.exp(-af)
    exp_ae = math.exp(-ae)
    frame_length = 10e4

    for i in range(3, len(sub2)):
        folder_path = sub2[i] 
        file_list = os.listdir(folder_path)
        sorted_file_list = sorted(file_list)
        for file_name in sorted_file_list:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):  
                file1.append(file_path)
        print(file1)
        label_dir = sub1[i]
        for j in range(len(file1)):
            idx = file_list[j]
            td = read_dataset(file1[j])
            # boxs = read_annotation(file2[j])
            # x_max, y_max, x_min, y_min = recover_bbox(boxs[0], boxs[1], boxs[2], boxs[3])
            # print(x_min, y_min, x_max, y_max)
            t_max = td.data.ts[-1]
            frame_start = td.data[0].ts
            frame_end = td.data[0].ts + frame_length
            # td_img = np.ones((td.height, td.width), dtype=np.uint8)
            frame_data = td.data[(td.data.ts >= frame_start) & (td.data.ts < frame_end)]

            valid_mask = (frame_data['x'] >= 0) & (frame_data['x'] < td.width) & (frame_data['y'] >= 0) & (frame_data['y'] < td.height)
            valid_data = frame_data[valid_mask]
            pixel_indices = valid_data['y'] * td.width + valid_data['x']
            unique_pixel_indices = np.unique(pixel_indices)
            polarity_dict = {pixel_idx: [] for pixel_idx in unique_pixel_indices}
         
            for datum in valid_data:
                pixel_idx = datum['y'] * td.width + datum['x']
                polarity_dict[pixel_idx].append(datum['p'])
        
            lengths = np.array([len(polarity_list) for polarity_list in polarity_dict.values()])
            max_length = lengths.max()
            for pixel_idx, polarity_list in polarity_dict.items():
                polarity_list += [polarity_list[-1]] * (max_length - len(polarity_list))

            sequence = np.zeros(max_length)
            wavename = "gaus1"
            img = np.zeros((td.height, td.width))
            video_real = np.zeros((3, max_length))
            result_img = np.zeros((td.height, td.width))

            for pixel_idx, polarity_list in polarity_dict.items():
                m = pixel_idx % td.width
                n = pixel_idx // td.width
                if 0 <= m < td.width and 0 <= n < td.height:
                    for k in range(max_length):
                        st1 = polarity_list[k]
                        x0 = exp_af * x0 + st1
                        y0 = exp_ae * y0 + ve * z0
                        z0 = 1 / (1 + math.exp(-(x0 - y0)))
                        sequence[k] = z0
                    cwtmatr, frequencies = pywt.cwt(sequence, np.arange(1, 4), wavename) 
                    len1, len2 = cwtmatr.shape  
                    for i1 in range(0, len1, 1):
                        for j1 in range(0, len2, 1):
                            video_real[i1, j1] = cwtmatr[i1, j1].real
                    img[n, m] = np.sum(video_real) 
                    if img[n, m] != 0: 
                        result_img[n, m] = 255
                    else:
                        result_img[n, m] = 0
            frame_path = os.path.join(label_dir, f"{idx}.jpg")
            print(frame_path)
            cv2.imwrite(frame_path,  result_img)  
            # print(f"Saved frame {idx} to {frame_path}")
        file1.clear()

asyncio.run(my_function())
