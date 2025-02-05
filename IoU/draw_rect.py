import cv2
import numpy as np
from imutils import perspective

def draw_rect(cnts, orig):
    for c in cnts:       
        if cv2.contourArea(c) < 100:
            continue
        box = cv2.minAreaRect(c)         
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2()else cv2.boxPoints(box)
    
        x_min, y_min = np.min(box, axis=0)
        x_max, y_max = np.max(box, axis=0)
        box = np.array(box, dtype="int")     
        box = perspective.order_points(box) 
        # cv2.drawContours(orig, [c.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)  
        # p = math.sqrt((y_max - y_min) ** 2 + (x_max - x_min) ** 2)
        p = (x_max - x_min) * (y_max - y_min)
        p = int(p)
        # p_dict = {}
        if p not in p_dict:
            p_dict[p] = []
        p_dict[p].append(box)

    return orig
