2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skindetect import SkinDetect
from gestures import Gestures
from camfeed import AndroidCamFeed

#______________________________________________________________
#setup capture
host = "10.42.0.128:8080"

## Create new AndroidCamFeed instance
acf = AndroidCamFeed(host)

calibration_counter = 0
calibration_interval = 30
accumulation_limit = 30
face_coords = []
gestures = Gestures()

print ("Get number of reps:")
reps = 0
reps_candidates = np.array([0]*6)
skindetect = SkinDetect()
while acf.isOpened() and (reps == 0):
    ## Read frame
    ret, frame = acf.read()
    if ret:
        if (calibration_counter % calibration_interval == 0):
            face_coords, success_flag = skindetect.set_skin_threshold_from_face(frame)
            if (not success_flag):
                calibration_counter-=1

        mask = skindetect.process(frame)
                
        if (calibration_counter %  calibration_interval == 0):        
            gestures.set_thresholds(face_coords)
            calibration_counter=0
        
        calibration_counter+=1
           
        frame, finger_count = gestures.process(frame, mask, face_coords)
        if (finger_count != -1):
            reps_candidates[finger_count] += 1
        
        if (np.amax(reps_candidates) > accumulation_limit):
            reps = np.argmax(reps_candidates)
        
        mask = cv2.resize(mask, (540, 960))        
        frame = cv2.resize(frame, (540, 960))
        #cv2.imshow('mask', cv2.bitwise_and(frame, frame, mask=mask))
        cv2.imshow('frame', frame)
   
print ("Number of reps: ", reps)

print ("Get number of sets:")
sets = 0
sets_candidates = np.array([0]*6)
skindetect = SkinDetect()
while acf.isOpened() and (sets == 0):
    ## Read frame
    ret, frame = acf.read()
    if ret:
        if (calibration_counter % calibration_interval == 0):
            face_coords, success_flag = skindetect.set_skin_threshold_from_face(frame)
            if (not success_flag):
                calibration_counter-=1

        mask = skindetect.process(frame)
                
        if (calibration_counter %  calibration_interval == 0):        
            gestures.set_thresholds(face_coords)
            calibration_counter=0
        
        calibration_counter+=1
           
        frame, finger_count = gestures.process(frame, mask, face_coords)
        if (finger_count != -1):
            reps_candidates[finger_count] += 1
        
        if (np.amax(sets_candidates) > accumulation_limit):
            sets = np.argmax(sets_candidates)
        
        mask = cv2.resize(mask, (540, 960))        
        frame = cv2.resize(frame, (540, 960))
        #cv2.imshow('mask', cv2.bitwise_and(frame, frame, mask=mask))
        cv2.imshow('frame', frame)

print ("Number of sets: ", sets)

# clean up
acf.release()
cv2.destroyAllWindows()
