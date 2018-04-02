2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skindetect import SkinDetect
from gestures import Gestures
from camfeed import AndroidCamFeed

CALIBRATION_INTERVAL = 30
ACCUMULATION_LIMIT = 10
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 2
FONT_THICKNESS = 3
FONT_COLOR = (0,255,0)
FONT_COLOR_2 = (0,255,255)
TEXT_POSITION_X = 0.10
TEXT_POSITION_X_2 = 0.30
TEXT_POSITION_Y = 0.80
VIEW_RESIZE = (540,960)
VIEW_WINDOW = 'frame'

face_coords = []
skindetect = None
gestures = None
    
def accumulate_finger_count_from_stream(stream, prompt):
    global face_coords, skindetect, gestures
    
    skindetect.reset_background()
    
    calibration_counter = 0
    finger_candidates = np.array([0]*6)
    num_fingers = -1
    while stream.isOpened() and (num_fingers == -1):
        ## Read frame
        ret, frame = stream.read()
        if ret:
            cv2.putText(frame,prompt,(int(TEXT_POSITION_X*frame.shape[1]),int(TEXT_POSITION_Y*frame.shape[0])),FONT,FONT_SCALE,FONT_COLOR,FONT_THICKNESS,8)
            
            if (calibration_counter % CALIBRATION_INTERVAL == 0):
                face_coords, success_flag = skindetect.set_skin_threshold_from_face(frame)
                if (not success_flag):
                    calibration_counter-=1
    
            mask = skindetect.process(frame)
                    
            if (calibration_counter %  CALIBRATION_INTERVAL == 0):        
                gestures.set_thresholds(face_coords)
                calibration_counter=0
            
            calibration_counter+=1
               
            frame, finger_count = gestures.get_finger_count(frame, mask, face_coords)
            
            if (finger_count != -1):
                cv2.putText(frame,str(finger_count),(int(TEXT_POSITION_X_2*frame.shape[1]),int(TEXT_POSITION_Y*frame.shape[0])),FONT,FONT_SCALE,FONT_COLOR_2,FONT_THICKNESS,8)
                finger_candidates[finger_count] += 1               
            
            if (np.amax(finger_candidates) > ACCUMULATION_LIMIT):
                num_fingers = np.argmax(finger_candidates)
                 
            frame = cv2.resize(frame, VIEW_RESIZE)
            cv2.imshow(VIEW_WINDOW, frame)
            
        if cv2.waitKey(1) == ord('q'):
            break
    
    return num_fingers

def wait_until_hand_is_down(stream):
    global face_coords, skindetect, gestures
    
    hand = 1
    while stream.isOpened() and (hand is not None):
        ret, frame = stream.read()
        if ret:
            cv2.putText(frame,"PUT HAND DOWN...",(int(TEXT_POSITION_X*frame.shape[1]),int(TEXT_POSITION_Y*frame.shape[0])),FONT,FONT_SCALE,FONT_COLOR,FONT_THICKNESS,8)
    
            mask = skindetect.process(frame)
               
            frame, hand, hand_crop = gestures.find_hand_contour(frame, mask, face_coords)
            frame = cv2.resize(frame, VIEW_RESIZE)
            cv2.imshow(VIEW_WINDOW, frame)
            
        if cv2.waitKey(1) == ord('q'):
            break

def wait_until_calibration(stream):
    global face_coords, skindetect, gestures
    success_flag = False
    while stream.isOpened() and not success_flag:
        ret, frame = stream.read()
        
        if ret:
            face_coords, success_flag = skindetect.set_skin_threshold_from_face(frame)
            cv2.putText(frame,"Look at camera...",(int(TEXT_POSITION_X*frame.shape[1]),int(TEXT_POSITION_Y*frame.shape[0])),FONT,FONT_SCALE,FONT_COLOR,FONT_THICKNESS,8)
            frame = cv2.resize(frame, VIEW_RESIZE)
            cv2.imshow(VIEW_WINDOW, frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
    gestures.set_thresholds(face_coords)

def wait_for_thumbs_up(stream):
    global face_coords, skindetect, gestures
    
    skindetect.reset_background()
    
    calibration_counter = 0
    thumbsup_accum = 0
    while stream.isOpened():
        ## Read frame
        ret, frame = stream.read()
        if ret:
            cv2.putText(frame,"SHOW THUMBS UP:",(int(TEXT_POSITION_X*frame.shape[1]),int(TEXT_POSITION_Y*frame.shape[0])),FONT,FONT_SCALE,FONT_COLOR,FONT_THICKNESS,8)
            
            if (calibration_counter % CALIBRATION_INTERVAL == 0):
                face_coords, success_flag = skindetect.set_skin_threshold_from_face(frame)
                if (not success_flag):
                    calibration_counter-=1
    
            mask = skindetect.process(frame)
                    
            if (calibration_counter %  CALIBRATION_INTERVAL == 0):        
                gestures.set_thresholds(face_coords)
                calibration_counter=0
            
            calibration_counter+=1
            
            is_thumbsup = gestures.is_thumbsup(frame, mask, face_coords)
            
            if is_thumbsup:
                thumbsup_accum+=1
                cv2.putText(frame,"THUMBS UP!",(int(TEXT_POSITION_X_2*frame.shape[1]),int(TEXT_POSITION_Y*frame.shape[0])),FONT,FONT_SCALE,FONT_COLOR_2,FONT_THICKNESS,8)
                
            if thumbsup_accum > ACCUMULATION_LIMIT:
                break;
                
            frame = cv2.resize(frame, VIEW_RESIZE)
            cv2.imshow(VIEW_WINDOW, frame)
            
        if cv2.waitKey(1) == ord('q'):
            break
        
#setup capture
host = "10.42.0.128:8080"

## Create new AndroidCamFeed instance
acf = AndroidCamFeed(host)
if (acf.isOpened()):
    gestures = Gestures()
    skindetect = SkinDetect()
    
    wait_until_calibration(acf)
    
    print ("Enter number of reps:")
    reps = accumulate_finger_count_from_stream(acf, "REPS?")
    print ("Number of reps is: ", reps)
    
    wait_until_hand_is_down(acf)
    
    print ("Enter number of sets:")
    sets = accumulate_finger_count_from_stream(acf, "SETS?")
    print ("Number of sets is: ", sets)
    
    wait_until_hand_is_down(acf)
    
    print ("Show thumbs up:")
    wait_for_thumbs_up(acf)

# clean up
acf.release()
cv2.destroyAllWindows()
