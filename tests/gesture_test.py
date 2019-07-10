2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from libs.skindetect import SkinDetect
from libs.gestures import Gestures
from libs.camfeed import AndroidCamFeed
import numpy as np

pause = False
# Handle a mouse click and output the colour under the click point
def click(event, x, y, flags, param):
    global pause
    if event == cv2.EVENT_LBUTTONUP:
        if not frame is None:
            ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
            print('YCrCB: ' + str(ycrcb_frame[y, x]))
            pause = True
            
cv2.namedWindow('mask')
cv2.namedWindow('frame')
cv2.setMouseCallback('mask', click)
cv2.setMouseCallback('frame', click)

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
    
    if not pause:    
        ## Read frame
        ret, frame = acf.read()
        backup=np.copy(frame)
        if ret:
            success = False
            if (calibration_counter % calibration_interval == 0):
                face_coords, success_flag = skindetect.set_skin_threshold_from_face(frame)
                if (not success_flag):
                    calibration_counter-=1
                else:
                    cv2.imwrite("frame.jpg", backup)
                    success = True
    
            mask = skindetect.process(frame)
            
            if (success):
                cv2.imwrite("mask.jpg", cv2.bitwise_and(backup, backup, mask=mask))
                    
            if (calibration_counter %  calibration_interval == 0):        
                gestures.set_thresholds(face_coords)
                calibration_counter=0
            
            calibration_counter+=1
               
            frame, gesture = gestures.get_finger_count(frame, mask, face_coords)
            
            mask = cv2.resize(mask, (540, 960))        
            frame = cv2.resize(frame, (540, 960))
            cv2.imshow('mask', cv2.bitwise_and(frame, frame, mask=mask))
            cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
       break
   
    if cv2.waitKey(1) == ord('w'):
       pause = False
    
# clean up
acf.release()
cv2.destroyAllWindows()
