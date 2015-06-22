# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:27:37 2015

@author: Arthur
"""

from cv2 import *
# initialize the camera
cam = VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
    imshow("cam-test",img)
    waitKey(0)
    destroyWindow("cam-test")
    imwrite("webcam.jpg",img) #save image