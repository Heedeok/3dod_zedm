import sys
import pyzed.sl as sl
import numpy as np
import cv2
from pathlib import Path
import enum


class AppType(enum.Enum):
    LEFT_AND_RIGHT = 1
    LEFT_AND_DEPTH = 2
    LEFT_AND_DEPTH_16 = 3
    ONLY_LEFT = 4
    ONLY_DEPTH = 5
    DEPTH_DATA = 6
    POINT_CLOUD = 7

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.2f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def main():
    
    if not sys.argv or len(sys.argv) != 4:
        sys.stdout.write("Usage: \n\n")
        sys.stdout.write("    ZED_SVO_Export A B C \n\n")
        sys.stdout.write("Please use the following parameters from the command line:\n")
        sys.stdout.write(" A - SVO file path (input) : \"path/to/file.svo\"\n")
        sys.stdout.write(" B - AVI file path (output) or image sequence folder(output) :\n")
        sys.stdout.write("         \"path/to/output/file.avi\" or \"path/to/output/folder\"\n")
        sys.stdout.write(" C - Export mode:  0=Export LEFT+RIGHT AVI.\n")
        sys.stdout.write("                   1=Export LEFT+DEPTH_VIEW AVI.\n")
        sys.stdout.write("                   2=Export LEFT+RIGHT image sequence.\n")
        sys.stdout.write("                   3=Export LEFT+DEPTH_VIEW image sequence.\n")
        sys.stdout.write("                   4=Export LEFT+DEPTH_16Bit image sequence.\n")
        sys.stdout.write("                   5=Export Only LEFT AVI.\n")
        sys.stdout.write("                   6=Export Only DEPTH AVI.\n")
        sys.stdout.write("                   7=Export DEPTH data(float32).\n")
        sys.stdout.write("                   8=Export POINTCLOUD data(RGBA float32). \n")
        sys.stdout.write(" A and B need to end with '/' or '\\'\n\n")
        sys.stdout.write("Examples: \n")
        sys.stdout.write("  (AVI LEFT+RIGHT):  ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/file.avi\" 0\n")
        sys.stdout.write("  (AVI LEFT+DEPTH):  ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/file.avi\" 1\n")
        sys.stdout.write("  (SEQUENCE LEFT+RIGHT):  ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\" 2\n")
        sys.stdout.write("  (SEQUENCE LEFT+DEPTH):  ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\" 3\n")
        sys.stdout.write("  (SEQUENCE LEFT+DEPTH_16Bit):  ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\""
                         " 4\n")
        exit()

    # Get input parameters
    svo_input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_as_video = True    
    app_type = AppType.LEFT_AND_RIGHT
    if sys.argv[3] == "1" or sys.argv[3] == "3":
        app_type = AppType.LEFT_AND_DEPTH
    if sys.argv[3] == "4":
        app_type = AppType.LEFT_AND_DEPTH_16
    if sys.argv[3] == "5":
        app_type = AppType.ONLY_LEFT
    if sys.argv[3] == "6":
        app_type = AppType.ONLY_DEPTH
    if sys.argv[3] == "7":
        app_type = AppType.DEPTH_DATA
        print('Export depth value (float32)')
    if sys.argv[3] == "8":
        app_type = AppType.POINT_CLOUD
        print('Export point cloud data (RGBA float32)')
    
    # Check if exporting to AVI or SEQUENCE
    if sys.argv[3] != "0" and sys.argv[3] != "1" and sys.argv[3] != "5" and sys.argv[3] != "6":
        output_as_video = False

    if not output_as_video and not output_path.is_dir():
        sys.stdout.write("Input directory doesn't exist. Check permissions or create it.\n",
                         output_path, "\n")
        exit()

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()
    
    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    print('SVO file fps : {}'.format(zed.get_camera_information().camera_fps))
    width = image_size.width
    height = image_size.height
    width_sbs = width * 2
    
    # Prepare side by side image container equivalent to CV_8UC4
    if app_type != AppType.ONLY_LEFT and app_type != AppType.ONLY_DEPTH: 
        svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)
    else: 
        svo_image_sbs_rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Prepare single image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_image = sl.Mat()
    point_cloud = sl.Mat()

    video_writer = None
    if output_as_video:

        if app_type != AppType.ONLY_LEFT and app_type != AppType.ONLY_DEPTH:    
            # Create video writer with MPEG-4 part 2 codec
            video_writer = cv2.VideoWriter(str(output_path),
                                        cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                        max(zed.get_camera_information().camera_fps, 25),
                                        (width_sbs, height))
        else : 
            print('We only use left side')
            # Create video writer with MPEG-4 part 2 codec
            video_writer = cv2.VideoWriter(str(output_path),
                                        cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                        max(zed.get_camera_information().camera_fps, 25),
                                        (width, height))

        if not video_writer.isOpened():
            sys.stdout.write("OpenCV video writer cannot be opened. Please check the .avi file path and write "
                             "permissions.\n")
            zed.close()
            exit()
    
    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.STANDARD # for computing
    # rt_param.sensing_mode = sl.SENSING_MODE.FILL # for displaying

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()

    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            if app_type == AppType.LEFT_AND_RIGHT:
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            elif app_type == AppType.LEFT_AND_DEPTH:
                zed.retrieve_image(right_image, sl.VIEW.DEPTH)
            elif app_type == AppType.LEFT_AND_DEPTH_16 :
                zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            elif app_type == AppType.ONLY_DEPTH:
                zed.retrieve_image(left_image, sl.VIEW.DEPTH)
            elif app_type == AppType.DEPTH_DATA:
                zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            elif app_type == AppType.POINT_CLOUD:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) # data type =F32_C4

            if output_as_video:

                if app_type != AppType.ONLY_LEFT and app_type != AppType.ONLY_DEPTH:

                    # Copy the left image to the left side of SBS image
                    svo_image_sbs_rgba[0:height, 0:width, :] = left_image.get_data()

                    # Copy the right image to the right side of SBS image
                    svo_image_sbs_rgba[0:, width:, :] = right_image.get_data()

                else :
                    svo_image_sbs_rgba[0:height, 0:width, :] = left_image.get_data()

                # Convert SVO image from RGBA to RGB
                ocv_image_sbs_rgb = cv2.cvtColor(svo_image_sbs_rgba, cv2.COLOR_RGBA2RGB)

                # Write the RGB image in the video
                video_writer.write(ocv_image_sbs_rgb)
            else:
                # Generate file names
                filename1 = output_path / ("left%s.png" % str(svo_position).zfill(6))
                filename2 = output_path / (("right%s.png" if app_type == AppType.LEFT_AND_RIGHT
                                           else "depth%s.png") % str(svo_position).zfill(6))
                filename3 = output_path / (("depth_data%s.txt") % str(svo_position).zfill(6))
                filename4 = output_path / (("pcd_data%s.bin") % str(svo_position).zfill(6))
               
                if app_type == AppType.LEFT_AND_DEPTH:
                    # Save Left images
                    cv2.imwrite(str(filename1), left_image.get_data())
                    # Save right images
                    cv2.imwrite(str(filename2), right_image.get_data()) 
                elif app_type == AppType.LEFT_AND_DEPTH_16 :
                    # Save Left images
                    cv2.imwrite(str(filename1), left_image.get_data())
                    # Save depth images (convert to uint16)
                    cv2.imwrite(str(filename2), depth_image.get_data().astype(np.uint16))
                elif app_type == AppType.DEPTH_DATA :
                    # Saver depth value (float 32)
                    fout = open(filename3, 'w')
                    np.savetxt(fout, depth_image.get_data())
                    fout.close()
                elif app_type == AppType.POINT_CLOUD :
                    # Saver pcd value - XYZA (A=color) (float 32)
                    point_cloud.get_data().tofile(filename4)


            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

    if output_as_video:
        # Close the video writer
        video_writer.release()

    zed.close()
    return 0


if __name__ == "__main__":
    main()