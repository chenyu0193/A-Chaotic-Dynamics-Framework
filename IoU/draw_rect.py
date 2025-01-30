import cv2
import numpy as np
from imutils import perspective

def draw_rect(cnts, orig):
    for c in cnts:         # 过滤点太小的轮廓点
        if cv2.contourArea(c) < 100:
            continue
        box = cv2.minAreaRect(c)           # 计算最小的外接矩形
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2()else cv2.boxPoints(box)
        # 找到左上角和右下角坐标
        x_min, y_min = np.min(box, axis=0)
        x_max, y_max = np.max(box, axis=0)
        box = np.array(box, dtype="int")      # 对轮廓点进行排序：左上、右上、右下、左下
        # print(box[0], box[2])
        box = perspective.order_points(box)  # 绘制轮廓
        # cv2.drawContours(orig, [c.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)  # 绘制轮廓
        # p = math.sqrt((y_max - y_min) ** 2 + (x_max - x_min) ** 2)
        p = (x_max - x_min) * (y_max - y_min)
        p = int(p)
        # p_dict = {}
        if p not in p_dict:
            p_dict[p] = []
        p_dict[p].append(box)

    return orig