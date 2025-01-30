import os
import numpy as np
import time
import cv2
import pywt
import math
import asyncio


class Timer(object):
    """Timer for making ad-hoc timing measurements"""
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.secs = 0
        self.msecs = 0
        self.start = 0
        self.end = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)

class Events(object):
    """
    Temporal Difference events.
    data: a NumPy Record Array with the following named fields
        x: pixel x coordinate, unsigned 16bit int
        y: pixel y coordinate, unsigned 16bit int
        p: polarity value, boolean. False=off, True=on
        ts: timestamp in microseconds, unsigned 64bit int
    width: The width of the frame. Default = 304.
    height: The height of the frame. Default = 240.
    """

    def __init__(self, num_events, width=304, height=240):
        """num_spikes: number of events this instance will initially contain"""
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.uint64)],
                                 shape=(num_events))
        self.width = width
        self.height = height

def load_atis_data(filename, flipX=0, flipY=0):
    """
    加载 ATIS 相机生成的二进制数据文件。

    参数：
        filename: str，文件路径
        flipX: int，是否在 X 轴翻转（默认值 0，表示不翻转）
        flipY: int，是否在 Y 轴翻转（默认值 0，表示不翻转）

    返回：
        td_data: dict，包含事件数据，字段为：
            - ts: 时间戳 (微秒)
            - x: x 坐标
            - y: y 坐标
            - p: 极性 (1 表示 ON，-1 表示 OFF)
    """
    td_data = {}

    with open(filename, 'rb') as f:
        # 解析文件头部
        header = []
        num_comment_lines = 0
        while True:
            pos = f.tell()
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if not line.startswith('%'):
                f.seek(pos)
                break
            words = line.split()
            if len(words) > 2:
                if words[1] == 'Date' and len(words) > 3:
                    header.append((words[1], f"{words[2]} {words[3]}"))
                else:
                    header.append((words[1], words[2]))
            num_comment_lines += 1

        # 读取事件类型和大小
        evType, evSize = 0, 8
        if num_comment_lines > 0:
            evType = int.from_bytes(f.read(1), byteorder='little')
            evSize = int.from_bytes(f.read(1), byteorder='little')

        # 计算事件数量
        bof = f.tell()
        f.seek(0, 2)  # 跳到文件末尾
        file_size = f.tell()
        num_events = (file_size - bof) // evSize

        # 读取时间戳和地址数据
        f.seek(bof, 0)  # 返回到事件开始处
        all_ts = np.fromfile(f, dtype=np.uint32, count=num_events)
        f.seek(bof + 4, 0)
        all_addr = np.fromfile(f, dtype=np.uint32, count=num_events)

    # 提取时间戳
    td_data['ts'] = all_ts.astype(np.float64)

    # 解析地址中的 x, y 和极性
    xmask = 0x00003FFF
    ymask = 0x0FFFC000
    polmask = 0x10000000
    xshift = 0
    yshift = 14
    polshift = 28

    all_addr = np.abs(all_addr)  # 确保地址非负
    td_data['x'] = ((all_addr & xmask) >> xshift)
    td_data['y'] = ((all_addr & ymask) >> yshift)
    td_data['p'] = -1 + 2 * ((all_addr & polmask) >> polshift).astype(np.float32)
    # (np.float64)
    # 翻转 X 和 Y 坐标
    if flipX > 0:
        td_data['x'] = flipX - td_data['x']
    if flipY > 0:
        td_data['y'] = flipY - td_data['y']

    return td_data

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
save_dir = "C:/Object Detection/N-CARS-Frames"
label_list = os.listdir("C:/Object Detection/N-CARS")
for j in range(len(label_list)):
    label = label_list[j]
    label_dir = os.path.join(save_dir, str(label))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if os.path.isdir(label_dir):
        sub1.append(label_dir)

sub2 = []
root_folder = "C:/Object Detection/N-CARS"
subfolders = os.listdir(root_folder)
# sorted_file_list = sorted(subfolders)
for subfolder in subfolders:
    # 使用os.path.join拼接完整的路径
    subfolder_path = os.path.join(root_folder, subfolder)
    # 判断是否为文件夹
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
    frame_length = 10e6

    for i in range(len(sub2)):
        folder_path = sub2[i]  # 替换为你的文件夹路径
        # 获取文件夹中的所有文件
        file_list = os.listdir(folder_path)
        sorted_file_list = sorted(file_list)
        # 遍历文件列表并逐个读取文件
        for file_name in sorted_file_list:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):  # 确保是文件而不是文件夹
                file1.append(file_path)
        print(file1)
        label_dir = sub1[i]
        for j in range(len(file1)):
            idx = file_list[j]
            td_data = load_atis_data(file1[j])
            num = len(td_data['x']) - 1

            filtered_data = {
                'x': [int(x) for x in td_data['x'][:num]],
                'y': [int(x) for x in td_data['y'][:num]],
                'p': [int(x) for x in td_data['p'][:num]]
            }

            def find_largest_under_100(numbers):
                # 使用列表推导式过滤出小于 100 的数字
                under_100 = [num for num in numbers if num < 100]

                # 如果列表不为空，返回最大值，否则返回 None
                return max(under_100) if under_100 else None

            width = find_largest_under_100(filtered_data['x']) + 1
            height = max(filtered_data['y']) + 1

            for i in range(num):
                if 0 <= filtered_data['x'][i] < width and 0 <= filtered_data['y'][i] < height:
                    # 将事件存储在对应像素的列表中
                    pixel_idx = filtered_data['y'][i] * width + filtered_data['x'][i]
                    if pixel_idx not in polarity_dict:
                        polarity_dict[pixel_idx] = []
                    polarity_dict[pixel_idx].append(filtered_data['p'][i])

            max_length = 30

            for pixel_idx, polarity_list in polarity_dict.items():
                # polarity_dict[pixel_idx] = str(polarity_dict[pixel_idx]).ljust(max_length, "0")
                num = polarity_list[len(polarity_list) - 1]
                polarity_list += [num] * (max_length - len(polarity_list))
                # print(f"Pixel ({pixel_idx % width}, {pixel_idx // width}): {polarity_list}")

            sequence = np.zeros(max_length)
            wavename = "gaus1"
            img = np.zeros((height, width))
            video_real = np.zeros((3, max_length))
            result_img = np.zeros((height, width))

            for pixel_idx, polarity_list in polarity_dict.items():
                m = pixel_idx % width
                n = pixel_idx // width
                if 0 <= m < width and 0 <= n < height:
                    for k in range(max_length):
                        st1 = polarity_list[k]
                        x0 = exp_af * x0 + st1
                        y0 = exp_ae * y0 + ve * z0
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
            frame_path = os.path.join(label_dir, f"{idx}.jpg")
            print(frame_path)
            cv2.imwrite(frame_path,  result_img)  # 将值缩放到 0-255 的范围
            # print(f"Saved frame {idx} to {frame_path}")
            polarity_dict.clear()
            length.clear()
        file1.clear()
asyncio.run(my_function())

