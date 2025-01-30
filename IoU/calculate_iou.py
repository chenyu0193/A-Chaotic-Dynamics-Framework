def calculate_iou(box1, box2):
    # box1 和 box2 格式：(x_min, y_min, x_max, y_max)

    # 计算相交矩形的坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 计算相交矩形的宽度和高度
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    # p...pp.print(inter_height, inter_width)
    # 计算相交面积
    intersection = inter_width * inter_height
    # 计算两个框的面积
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union = area_box1 + area_box2 - intersection
    # 计算 IoU
    iou = intersection / union
    # print(intersection, union)
    return iou