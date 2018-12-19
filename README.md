# yzy_vo
An easy-to-understand visual odometry project based on OpenCV 3.

Accept web camera or mp4 video file as input.

A test.mp4 video file include for convenience. This file was obtained from https://www.youtube.com/watch?v=Lgy1F72Oi2g

Build:

    cd build/
    rm -rf *
    cmake ..
    make


Useage:

    ./yzy_vo cam [cameraIdex]
    ./yzy_vo video [pathTovideo]
