import numpy as np

def read_annotation(filename):
    '''
    with open(filename, 'rb') as f:
        rows = np.frombuffer(f.read(2), dtype=np.int16)[0]
        cols = np.frombuffer(f.read(2), dtype=np.int16)[0]
        box_contour = np.frombuffer(f.read(rows*cols*2), dtype=np.int16)
        # box_contour = np.reshape(box_contour, (rows, cols))

        rows = np.frombuffer(f.read(2), dtype=np.int16)[0]
        cols = np.frombuffer(f.read(2), dtype=np.int16)[0]
        obj_contour = np.frombuffer(f.read(rows*cols*2), dtype=np.int16)
        obj_contour = np.reshape(obj_contour, (rows, cols))
    # return box_contour, obj_contour
    '''
    f = open(filename)
    box_contour = np.fromfile(f, dtype=np.int16)
    box_contour = np.array(box_contour[2:10])
    f.close()
    bbox = np.array([
            box_contour[0], box_contour[1], 
            box_contour[2] - box_contour[0],  
            box_contour[5] - box_contour[1],  
        ])
    # bbox[:2] = np.maximum(bbox[:2], 0)
    return bbox  
