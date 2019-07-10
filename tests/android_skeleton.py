#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from libs.skeleton_detect import SkeletonDetect
from libs.camfeed import AndroidCamFeed

#______________________________________________________________
#setup capture
host = "10.42.0.128:8080"

## Create new AndroidCamFeed instance
acf = AndroidCamFeed(host)
skeleton_detect = SkeletonDetect()

# Handle a mouse click and output the colour under the click point
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        if not frame is None:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
            print('HSV: ' + str(hsv_frame[y, x]))

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', click)

# capture loop
while acf.isOpened():
    ## Read frame
    ret, frame = acf.read()
    if ret:        
        
        mask = skeleton_detect.process(frame)
        
        mask = cv2.resize(mask, (540, 960))        
        frame = cv2.resize(frame, (540, 960))
        cv2.imwrite("skeleton.jpg", frame)
        cv2.imshow('mask', cv2.bitwise_and(frame, frame, mask=mask))

    if cv2.waitKey(1) == ord('q'):
       break
    
# clean up
acf.release()
cv2.destroyAllWindows()
