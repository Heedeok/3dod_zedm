import sys
import pyzed.sl as sl
import numpy as np
import cv2
import os
import re
import pypcd

width = 1920
height = 1080

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.2f%s' % (bar, percent_done, '%'))
    sys.stdout.flush()

def extract_center_coordi_basic(center_index, depth_map):
    ''' Extract center index which has max depth in depth map
        in this function, depth map is target depth map 
    '''

    frame_idx, x1, y1, x2, y2 = center_index

    box = depth_map[x1 : x2, y1 : y2]
    max_depth = np.max(box)
    max_index = np.unravel_index(box.argmax(), box.shape)
    min_depth = np.min(box)
    min_index = np.unravel_index(box.argmin(), box.shape)

    center_x = (x1 + x2) // 2 
    center_y = (y1 + y2) // 2
    center_z = (max_depth + min_depth) / 2
    
    return center_x, center_y, center_z 

def transform_box2D_to_index(target_coordi):
    ''' Transform target coordi (x,y (0~1 float32)) to numpy index and frame number
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
            coordi_index.extend([count,0,0,height-1, width-1])
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

def transform_pcd_bin_to_pcl_format(pcd_data_file):
    pcd_data = pypcd.PointCloud.from_path(pcd_data_file)
    pcd_data.pc_data['x'] -= pcd_data.pc_data['x'].mean()
    pcd_data.save_pcd('./testfile/test.pcd', compression='binary_compressed')

    pcd_data = np.fromfile('./testfile/test.pcd', dtype=np.float32)
    return pcd_data



def main():
   
    # depth_float32 = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/depthfile/depth_data000001.txt"
    # target_coordi = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/avifile/test_target_coordi.txt"
    # left_img = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/pngfile/left000001.png"   
    liadr_data = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/pcdfile/pcd_data000001.bin"
    test_data = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/LidarObstacleDetection/src/sensors/data/pcd/data_1/0000000000.pcd"      

    # depth_float32 = np.loadtxt(depth_float32, dtype=np.float32)
    # left_img = cv2.imread(left_img, cv2.IMREAD_UNCHANGED)
    pcd_data = np.fromfile(liadr_data, dtype=np.float32)
    test_data = np.fromfile(test_data, dtype=np.float32)

    print('====Zed depth=====')
    print('lidar type : {}'.format(pcd_data.dtype))
    print('lidar shape : {}'.format(pcd_data.shape))
    # pcd_data = pcd_data.reshape(height, width, 4)
    # print('lidar shape : {}'.format(pcd_data.shape))
    print('max : {}, min : {}'.format(pcd_data.max(), pcd_data.min()))
    # pcd_data = pcd_data.astype(np.float64)
    # print(pcd_data.dtype)
    # print(pcd_data[3])
    print('[0]:{}, [1]:{}, [2]:{}, [3]:{}'.format(pcd_data[0], pcd_data[1], pcd_data[2], pcd_data[3]))
    nan = pcd_data[0]
    if pcd_data[0]==nan:
        pcd_data[0] = 0.0
    print(pcd_data[0])
    # pcd_data = transform_pcd_bin_to_pcl_format(pcd_data)
    # print('new lidar type : {}'.format(pcd_data.dtype))
    # print('new lidar shape : {}'.format(pcd_data.shape))
    # print('max : {}, min : {}'.format(pcd_data.max(), pcd_data.min()))


    print('=====Lidar obstacle====')
    print('lidar type : {}'.format(test_data.dtype))
    print('lidar shape : {}'.format(test_data.shape))
    print('max : {}, min : {}'.format(test_data.max(), test_data.min()))
    print('[0]:{}, [1]:{}, [2]:{}, [3]:{}'.format(test_data[-4], test_data[-3], test_data[-2], test_data[-1]))
    print(test_data[-1].dtype)
















    # extract box index
    # center_index = transform_box2D_to_index(target_coordi)
    
    # frame_idx, x1, y1, x2, y2 = center_index[0]

    # # extract center coordinate
    # center_x, center_y, center_z = extract_center_coordi_basic(center_index[0], depth_float32)

    # cut_img = np.zeros((1080, 1920,4)).astype(np.uint8)
    # cut_img[x1 : x2, y1 : y2, :] = left_img[x1 : x2, y1 : y2, :]

    # cut_img[center_x, center_y,:] = [255, 255, 255, 0]

    # cv2.imshow('Unchange', cut_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()