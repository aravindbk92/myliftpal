#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from skeleton_detect import SkeletonDetect
from camfeed import AndroidCamFeed
import matplotlib.pyplot as plt

#______________________________________________________________
#setup capture
host = "10.42.0.128:8080"

## Create new AndroidCamFeed instance
acf = AndroidCamFeed(host)
skindetect = SkinDetect()
gestures = Gestures()

calibration_counter = 0
calibration_interval = 30
face_coords = []
# capture loop
while acf.isOpened():
    ## Read frame
    ret, frame = acf.read()
    if ret:        
        
        mask = skeleton_detect.process(frame)
        
        mask = cv2.resize(mask, (540, 960))        
        frame = cv2.resize(frame, (540, 960))
        cv2.imshow('mask', cv2.bitwise_and(frame, frame, mask=mask))

    if cv2.waitKey(1) == ord('q'):
       break
    
# clean up
acf.release()
cv2.destroyAllWindows()
