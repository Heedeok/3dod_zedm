import sys
import pyzed.sl as sl
import numpy as np
import cv2
from PIL import Image # image to numpt array
from matplotlib import image
from matplotlib import pyplot

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.2f%s' % (bar, percent_done, '%'))
    sys.stdout.flush()

def main():
    print('hello')
    depth_img_uint16 = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/pngfile/depth000001.png"
    depth_img_float32 = "/home/iasl/object_detection/yolo_tf2/3dod_zedm/depthfile/depth_data000001.png"
    
    img_uint16 = cv2.imread(depth_img_uint16, cv2.IMREAD_ANYDEPTH)
    img_float32 = cv2.imread(depth_img_float32, cv2.IMREAD_ANYDEPTH)
    print('shape uint16 : {}'.format(img_uint16.shape))
    print('type uint16 : {}'.format(img_uint16.dtype))
    print('img_uint16[100][100] : {}'.format(img_uint16[100][100]))
    print('shape float32 : {}'.format(img_float32.shape))
    print('type float32 : {}'.format(img_float32.dtype))
    print('img_float32[100][100] : {}'.format(img_float32[100][100]))
    # for x in range(100):
    #     for y in range(400):
    #         img_cv2[x][y]=0
    # cv2.imshow('result',img_cv2)
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()