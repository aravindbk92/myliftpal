2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skindetect import SkinDetect
from gestures import Gestures
from camfeed import AndroidCamFeed
from ar_marker import ARMarker

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
ar_marker = None

def display_text(frame, text,  color=FONT_COLOR, x_pos=TEXT_POSITION_X, y_pos=TEXT_POSITION_Y):
    cv2.putText(frame,text,(int(x_pos*frame.shape[1]),int(y_pos*frame.shape[0])),FONT,FONT_SCALE,FONT_COLOR,FONT_THICKNESS,8)
    
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
            display_text(frame, prompt)
            
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
                display_text(frame, str(finger_count), FONT_COLOR_2, TEXT_POSITION_X_2)
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
            display_text(frame, "PUT HAND DOWN...",)
    
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
            display_text(frame, "Look at camera...")
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
            display_text(frame, "Was this weight ok?")
            
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
                display_text(frame, "THUMBS UP!", FONT_COLOR_2, TEXT_POSITION_X_2)
                
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
    ar_marker = ARMarker()
    
    wait_until_calibration(acf)
    
    print ("Enter number of reps:")
    reps = accumulate_finger_count_from_stream(acf, "REPS?")
    print ("Number of reps is: ", reps)
    
    wait_until_hand_is_down(acf)
    
    print ("Enter number of sets:")
    sets = accumulate_finger_count_from_stream(acf, "SETS?")
    print ("Number of sets is: ", sets)
    
    wait_until_hand_is_down(acf)
    
    print ("Finding initial barbell position and identifying exercise...")
    #display_text(frame, "Finding initial barbell position...")
    #barbell_position, exercise = ar_marker.find_initial_barbell_position(frame)
    
    ### Loop for detection ###
    #ar_marker.set_deadlift_rep_threshold(knee_marker_y) #Set y of knee position (so if barbell goes above knee it is counted as a rep)
    
    ### Loop for set up stage ###
    
    ### Loop while lifting - count reps here ###
    # barbell_position = ar_marker.get_marker_center(frame)
    # current_rep = ar_marker.count_reps(barbell_position)
    # ar_marker.track_marker(barbell_position)
    # ar_marker.marker_history has history of barbell positions
    
    print ("Is this weight ok?")
    wait_for_thumbs_up(acf)
    
    ar_marker.reset()

# clean up
acf.release()
cv2.destroyAllWindows()
