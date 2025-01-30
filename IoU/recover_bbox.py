def recover_bbox(x_min, y_min, w, h):
    # Calculate the right-bottom coordinates of the bounding box
    x_max = int(x_min + w)
    y_max = int(y_min + h)

    # Calculate the width and height of the bounding box
    x = int(x_min)
    y = int(y_min)

    return x_max, y_max, x, y