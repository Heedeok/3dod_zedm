import sys
import pyzed.sl as sl
import numpy as np
import cv2

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.2f%s' % (bar, percent_done, '%'))
    sys.stdout.flush()



def main():
    depth_uint16 = "/home/jwk6844/objecet_detection/3dod_zedm/pngfile/depth000001.png"
    depth_float32 = "/home/jwk6844/objecet_detection/3dod_zedm/depthfile/depth_data000001.txt"
    target_coordi = "/home/jwk6844/objecet_detection/3dod_zedm/avifile/test_target_coordi.txt"
    left_img = "/home/jwk6844/objecet_detection/3dod_zedm/pngfile/left000001.png"   

    depth_uint16 = cv2.imread(depth_uint16, cv2.IMREAD_ANYDEPTH)
    depth_float32 = np.loadtxt(depth_float32, dtype=np.float32)
    left_img = cv2.imread(left_img, cv2.IMREAD_UNCHANGED)

    print('shape depth_uint16 : {}'.format(depth_uint16.shape))
    print('type depth_uint16 : {}'.format(depth_uint16.dtype))
    print('depth_uint16[100][100] : {}\n'.format(depth_uint16[100][100]))

    print('shape depth_float32 : {}'.format(depth_float32.shape))
    print('type depth_float32 : {}'.format(depth_float32.dtype))
    print('depth_float32[100][100] : {}\n'.format(depth_float32[100][100]))

    print('shape left_img : {}'.format(left_img.shape))
    print('type left_img : {}'.format(left_img.dtype))
    print('left_img[100][100] : {}\n'.format(left_img[100][100]))


    files = open(target_coordi, 'r')
    lines = files.readlines()
    count =0 
    for line in lines:
        # print('line{} : {}'.format(count, line.strip()))
        # print(line.split())
        if count == 0:
            x1 = line.split()[1]
            if x1[0] == '[':
                x1 = x1[1:]
            y1 = line.split()[2]
            x2 = line.split()[3]
            y2 = line.split()[4]
            if y2[-1] ==']':
                y2 = y2[:len(y2)-1]
        count +=1
    
    y1_ind = int(float(x1)*1920)
    y2_ind = int(float(x2)*1920)
    x1_ind = int(float(y1)*1080)
    x2_ind = int(float(y2)*1080)-1
    # print('x1 : {} -> y1 index : {} '.format(x1,x1_ind))
    # print('x2 : {} -> y2 index : {} '.format(x2,x2_ind))
    # print('y1 : {} -> x1 index : {} '.format(y1,y1_ind))
    # print('y2 : {} -> x2 index : {}\n '.format(y2,y2_ind))  

    cut_img = np.zeros((1080, 1920,4)).astype(np.uint8)
    cut_img[x1_ind:x2_ind, y1_ind:y2_ind, :] = left_img[x1_ind:x2_ind, y1_ind:y2_ind, :]



    box_uint16 = depth_uint16[x1_ind:x2_ind,y1_ind:y2_ind]
    box_float32 = depth_float32[x1_ind:x2_ind,y1_ind:y2_ind]

    max_depth_uint16 = np.max(box_uint16)
    max_depth_float32 = np.max(box_float32)
    max_index_uint16 = np.unravel_index(box_uint16.argmax(), box_uint16.shape)
    max_index_float32 = np.unravel_index(box_float32.argmax(), box_float32.shape)
    
    min_depth_uint16 = np.min(box_uint16)
    min_depth_float32 = np.min(box_float32)
    min_index_uint16 = np.unravel_index(box_uint16.argmin(), box_uint16.shape)
    min_index_float32 = np.unravel_index(box_float32.argmin(), box_float32.shape)

    print('max uint16 : {}, min uint16 : {} '.format(max_depth_uint16, min_depth_uint16))
    print('max index : {}, min index : {} '.format(max_index_uint16, min_index_uint16))
    print('max float32 : {}, min float32 : {} '.format(max_depth_float32, min_depth_float32))
    print('max index : {}, min index : {} \n'.format(max_index_float32, min_index_float32))

    center_uint16_x = (x1_ind + x2_ind) // 2 
    center_uint16_y = (y1_ind + y2_ind) // 2
    center_uint16_z = (max_depth_uint16 + min_depth_uint16) / 2 

    center_float32_x = (x1_ind + x2_ind) // 2 
    center_float32_y = (y1_ind + y2_ind) // 2
    center_float32_z = (max_depth_float32 + min_depth_float32) / 2

    print('Center uint16 : ({},{}), depth : {}mm'.format(center_uint16_x, center_uint16_y, center_uint16_z))
    print('Center float32 : ({},{}), depth : {}mm'.format(center_float32_x, center_float32_y, center_float32_z))

    cut_img[center_float32_x, center_float32_y,:] = [255, 255, 255, 0]




    cv2.imshow('Unchange', cut_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()