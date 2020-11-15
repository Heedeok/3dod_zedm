import sys
import pyzed.sl as sl
from signal import signal, SIGINT
import cv2

## python video.py [outputfile.svo]


cam = sl.Camera()

def handler(signal_received, frame):
    cam.disable_recording()
    cam.close()
    sys.exit(0)

signal(SIGINT, handler)

def main():
    if not sys.argv or len(sys.argv) != 2:
        print("Only the path of the output SVO file should be passed as argument.")
        exit(1)

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080 # max fps 30
    # init.camera_resolution = sl.RESOLUTION.HD720 # max fps 60
    # init.camera_resolution = sl.RESOLUTION.VGA # max fps 100
    init.depth_stabilization = False # to improve computational performance
    init.depth_mode = sl.DEPTH_MODE.ULTRA # Use ULTRA depth mode
    init.coordinate_units = sl.UNIT.MILLIMETER # Use millimeter units (for depth measurements)

    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.FILL
    img = sl.Mat() 

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)
   
    print('If you want to Recording, press "r"')
    while True:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(img, sl.VIEW.LEFT)
            cv2.imshow('result', img.get_data())
            key = cv2.waitKey(20)
        
        if key == 114:
            break

    if key == 114:

        path_output = sys.argv[1] # filename.svo 형식으로 작성
        recording_param = sl.RecordingParameters(path_output, sl.SVO_COMPRESSION_MODE.LOSSLESS)
        err = cam.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            exit(1)

        print('SVO file is Recording, use Ctrl-C to stop')
        frames_recorded = 0
        while True:
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(img, sl.VIEW.LEFT)
                cv2.imshow('result', img.get_data())
                key_video = cv2.waitKey(20)
                frames_recorded += 1
                print("Frame count: " + str(frames_recorded), end="\r")

if __name__ == "__main__":
    main()