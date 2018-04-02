#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 03:18:06 2018

@author: abk
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ar_marker import ARMarker

VIEW_RESIZE = (540,960)
VIEW_WINDOW = 'frame'

ar_marker = ARMarker()
vidcap = cv2.VideoCapture('office_deadlift.avi')
while (vidcap.isOpened()):
    ## Read frame
    vid_read_success,frame = vidcap.read()
    center =  ar_marker.get_marker_center(frame)
    cv2.circle(frame,center,10,(255,255,255),2)
    frame = cv2.resize(frame, VIEW_RESIZE)
    cv2.imshow(VIEW_WINDOW, frame)
            
    if cv2.waitKey(1) == ord('q'):
        break
    
vidcap.release()
cv2.destroyAllWindows()