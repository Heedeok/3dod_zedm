# ZED api for capturing and recording images and depth.

This repo provides a implementation of ZED mini api

## Usage

#### image_capture.py

```bash
python image_capture.py
```
press "r" for catpure the image
(file path : ./pngfile/xxxx.png)

#### video_capture.py

```bash
python video_capture.py [outputfile.svo]
```
press "r" for recording the image
press "q" to stop recording
(output file name must be xxxx.svo)

#### convert_svo_into_avi.py

```bash
python convert_svo.py [inpu/svofile.svo] [output/avifile.avi or directory] [mode]

# 5, 6 mode added for left side image
# 5 - only using left image, 6 - only using depth image
# 7, 8 mode added for export depthe and point cloud data
# depth data : float32, (height, width)
# pcd data : float32, (height, width, xyza # a is color of pixel)
```
##### load pcd file example
```bash
import numpy as np
np.array([1,2,3]).tofile("a.bin")
print np.fromfile("a.bin", dtpye=np.float32)
```

## Test from ZED mini


| svo file   | video mode | frame per second | output resolution |
|-------------|---------|---------|---------|
| test.svo    | 1080p  | 30   | 3840*1080  |
| test1.svo | 720p   | 60    | 2560*720   |


## References

I had to reference the official repos to piece together the complete picture.

- https://www.stereolabs.com/docs/video/using-video/
    - official STEREOLABS API documentation
- https://github.com/stereolabs/zed-examples
    - official STEREOLABS github repo for api








