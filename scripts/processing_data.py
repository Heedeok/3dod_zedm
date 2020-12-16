import sys
import pyzed.sl as sl
import numpy as np
import cv2
import os
import re
import pypcd
import math

'''for check nan float
    math.isnan(object)'''

width = 1920
height = 1080

def extract_max_and_min_index(x1, y1, x2, y2, depth_map):
    '''Extract max and min index for numpy ndarray'''

    target_map = depth_map[x1 : x2, y1 : y2]

    # axis=1 means the max index along x line
    max_xline = np.nanargmax(target_map, axis =1)
    min_xline = np.nanargmin(target_map, axis =1)

    tmp = 0 
    max_index = [0, 0]
    min_index = [0, 0]

    for i, ind in enumerate(max_xline):
        if target_map[i][ind] >= tmp:
            max_index[0] = i + x1
            max_index[1] = ind + y1

    tmp = target_map[max_index[0]-x1][max_index[1]-y1]

    for i, ind in enumerate(min_xline):
        if target_map[i][ind] <= tmp:
            min_index[0] = i + x1
            min_index[1] = ind + y1

    return max_index, min_index

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.2f%s' % (bar, percent_done, '%'))
    sys.stdout.flush()

def extract_center_coordi_basic(center_index, depth_map):
    ''' Extract center index which has max depth in depth map
        in this function, depth map is target 2D boxed depth map 
    '''

    frame_idx, x1, y1, x2, y2 = center_index

    max_index, min_index = extract_max_and_min_index(x1, y1, x2, y2, depth_map)

    max_depth = depth_map[max_index[0]][max_index[1]]
    min_depth = depth_map[min_index[0]][min_index[1]]
    
    center_depth = (max_depth + min_depth)/2
    obj_width = y2 -y1
    obj_height = x2 -x1

    return center_depth, obj_height, obj_width

def transform_box2D_to_index(target_coordi):
    ''' Transform target coordi file (frame number ,x1, y1, x2, y2) to numpy index
        frame_number, x1_ind, y1_ind, x2_ind, y2_ind
    '''
    
    # load target_coordinates 
    target_coordi = open(target_coordi, 'r')
    lines = target_coordi.readlines()
    coordi_index = []
    count = 0 
    for line in lines:
        
        # target 이 비어있는 경우 고려해주어야 함
        if count != int(line.split()[0]):
            coordi_index.extend([count,0,0,height-1, width-1]) # if frame empty, return entire pixel index
            count += 1
            continue

        x1 = line.split()[1]
        if x1[0] == '[':
            x1 = x1[1:]
        y1 = line.split()[2]
        x2 = line.split()[3]
        y2 = line.split()[4]
        if y2[-1] ==']':
            y2 = y2[:len(y2)-1]

        y1_ind = int(float(x1)*width)
        y2_ind = int(float(x2)*width)
        x1_ind = int(float(y1)*height)
        x2_ind = int(float(y2)*height)

        if x2_ind == height:
            x2_ind -= 1
        if y2_ind == height:
            y2_ind -= 1

        coordi_index.extend([count, x1_ind, y1_ind, x2_ind, y2_ind])
        count += 1
    
    return np.array(coordi_index).reshape(len(coordi_index)//5,5) 

def extract_3d_object_detection_basic():
    return True


def main():
   
    depth_data = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/depthfile/depth_data000001.txt"
    target_coordi = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/avifile/test_target_coordi.txt"
    left_img = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/pngfile/left000001.png"   
    # liadr_data = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/pcdfile/pcd_data000001.bin"    

    depth_data = np.loadtxt(depth_data, dtype=np.float32)
    left_img = cv2.imread(left_img, cv2.IMREAD_UNCHANGED)
    # pcd_data = np.fromfile(liadr_data, dtype=np.float32)


    print('=====Zed depth====')
    print('deptrh type : {}'.format(depth_data.dtype))
    print('lidar shape : {}'.format(depth_data.shape))
    print('max : {}, min : {}'.format(depth_data.max(), depth_data.min()))
    print('depth[100][100] : {}'.format(depth_data[100][100]))
    # print(depth_data)



    # extract box index
    center_index = transform_box2D_to_index(target_coordi)
    
    frame_idx, x1, y1, x2, y2 = center_index[0]
    print(center_index[0])

    # extract center coordinate
    center_depth, obj_height, obj_width = extract_center_coordi_basic(center_index[0], depth_data)
    print(center_depth, obj_height, obj_width)

    cut_img = np.zeros((1080, 1920,4)).astype(np.uint8)
    cut_img[x1 : x2, y1 : y2, :] = left_img[x1 : x2, y1 : y2, :]

    cut_img[ (x2+x1)//2, (y2+y1)//2,:] = [0, 0, 255, 0]

    cv2.imshow('Unchange', cut_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()