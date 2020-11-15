import pyzed.sl as sl
import cv2
from absl import app, flags, logging
from absl.flags import FLAGS

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.camera_fps = 50  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Capture 50 frames and stop
    i = 0
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    print('If you want capture the image, press "r"')
    while True:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv2.imshow('result', image.get_data())
            key=cv2.waitKey(1000//init_params.camera_fps)
            if key == 114 :
                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
                cv2.imwrite('./pngfile/{}.png'.format(timestamp.get_milliseconds()%10000),image.get_data())
                print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
                  timestamp.get_milliseconds()))
                print('Image saved to ./pngfile/{}.png'.format(timestamp.get_milliseconds()%10000))
                break
            
    # Close the camera
    zed.close()

if __name__ == "__main__":
    
    main()
