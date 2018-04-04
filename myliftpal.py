2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skindetect import SkinDetect
from gestures import Gestures
from camfeed import AndroidCamFeed
from ar_marker import ARMarker
from skeleten import Skeleton
from ar_marker import ARMarker
from Point import point
import simpleaudio as sa

CALIBRATION_INTERVAL = 30
ACCUMULATION_LIMIT = 10
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 2
FONT_THICKNESS = 3
FONT_COLOR = (0,255,0)
FONT_COLOR_2 = (0,255,255)
TEXT_POSITION_X = 0.10
TEXT_POSITION_X_2 = 0.60
TEXT_POSITION_Y = 0.80
VIEW_RESIZE = (540,960)
VIEW_WINDOW = 'frame'

face_coords = []

skindetect = None
gestures = None
ar_marker = None

skeleton = Skeleton()

isFirst = True
liftingStage = True
setUpStage = True

lower_threshold_ycrcb = [1, 60, 120]
upper_threshold_ycrcb = [200, 118, 135]

def apply_mask(frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(img_ycrcb)
    y = clahe.apply(y)
    img_enhanced_ycrcb = cv2.merge((y,cr,cb))
    
    mask_ycrcb = cv2.inRange(img_enhanced_ycrcb, np.array(lower_threshold_ycrcb), np.array(upper_threshold_ycrcb))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_ycrcb = cv2.morphologyEx(mask_ycrcb, cv2.MORPH_CLOSE, kernel, iterations=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_ycrcb = cv2.morphologyEx(mask_ycrcb, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return mask_ycrcb

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
                wave_obj = sa.WaveObject.from_wave_file("audio/keepWeight.wav")
                play_obj = wave_obj.play()
            if thumbsup_accum > ACCUMULATION_LIMIT:
                wave_obj = sa.WaveObject.from_wave_file("audio/reduceWight.wav")
                play_obj = wave_obj.play()
                break;
            frame = cv2.resize(frame, VIEW_RESIZE)
            cv2.imshow(VIEW_WINDOW, frame)
            
        if cv2.waitKey(1) == ord('q'):
            break            
            
#setup capture
host = "10.42.0.128:8080"

welcome_start = True
reps_start = True
hand_down_one = True
hand_down_two = True
sets_start = True
marker_init = True
check_weight = True

## Create new AndroidCamFeed instance
acf = AndroidCamFeed(host)
if (acf.isOpened()):
    gestures = Gestures()
    skindetect = SkinDetect()
    ar_marker = ARMarker()
    if(welcome_start):
            welcome_start = False
            wave_obj = sa.WaveObject.from_wave_file("audio/Welcome.wav")
            play_obj = wave_obj.play()
    wait_until_calibration(acf)
    
    print ("Enter number of reps:")
    if(reps_start):
            reps_start = False
            wave_obj = sa.WaveObject.from_wave_file("audio/howManyreps.wav")
            play_obj = wave_obj.play()
    reps = accumulate_finger_count_from_stream(acf, "REPS?")
    print ("Number of reps is: ", reps)
    if(hand_down_one):
            hand_down_one = False
            wave_obj = sa.WaveObject.from_wave_file("audio/handDown.wav")
            play_obj = wave_obj.play()
    wait_until_hand_is_down(acf)
    
    if(sets_start):
            sets_start = False
            wave_obj = sa.WaveObject.from_wave_file("audio/sets_start.wav")
            play_obj = wave_obj.play()
    print ("Enter number of sets:")
    sets = accumulate_finger_count_from_stream(acf, "SETS?")
    print ("Number of sets is: ", sets)
    if(hand_down_two):
            hand_down_two = False
            wave_obj = sa.WaveObject.from_wave_file("audio/handDown.wav")
            play_obj = wave_obj.play()
    wait_until_hand_is_down(acf)
    
    
    
    print ("Finding initial barbell position and identifying exercise...")
    #display_text(frame, "Finding initial barbell position...")
    
    while(acf.isOpened()):
        ret, frame = acf.read()
        
        if(not ret):
            continue
          
        if(marker_init):
            marker_init = False
            barbell_position, exercise = ar_marker.find_initial_barbell_position(frame)
            ar_marker.set_deadlift_rep_threshold(940*(3/4)) #Set y of knee position (so if barbell goes above knee it is counted as a rep)
        else:
            barbell_position = ar_marker.get_marker_center(frame)     
        
        barbellPt = point(int(barbell_position[0]),int(barbell_position[1]))
        
          
        
        mask = apply_mask(frame)    
        
        #find all contours in the screen
        im2,contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        labeled_frame , contourskeleton = skeleton.delect_skeleton(frame, contours)
    
        labeled_frame = skeleton.draw_skeleton(labeled_frame)    
        
        #draw all contours in the the frame so they are visible
        #cv2.drawContours(labeled_frame, contours, -1, (255,255,255), 1)
        
        if(not skeleton.setup_metrics(labeled_frame,barbellPt)):
            cv2.putText(frame,'Setup Stage',(int(TEXT_POSITION_X_2*frame.shape[1]),int(TEXT_POSITION_Y*frame.shape[0])),FONT,FONT_SCALE,FONT_COLOR_2,FONT_THICKNESS,8)
            if(setUpStage):
                setUpStage = False
                wave_obj = sa.WaveObject.from_wave_file("audio/setup.wav")
                play_obj = wave_obj.play()
        else:
            skeleton.lifting_metrics(labeled_frame,barbellPt)
            cv2.putText(frame,'Lifting Stage',(int(TEXT_POSITION_X_2*frame.shape[1]),int(TEXT_POSITION_Y*frame.shape[0])),FONT,FONT_SCALE,FONT_COLOR_2,FONT_THICKNESS,8)
            if(liftingStage):
                liftingStage = False
                wave_obj = sa.WaveObject.from_wave_file("audio/lift.wav")
                play_obj = wave_obj.play()
        
        current_rep = ar_marker.count_reps(barbell_position)
        if(current_rep == 0):
            break
        ar_marker.track_marker(barbell_position)
        for pos in ar_marker.marker_history:
            cv2.circle(frame,pos,5,(0,255,0),3)
        #ar_marker.marker_history has history of barbell positions
        
        mask = cv2.resize(mask, (540, 960))        
        labeled_frame = cv2.resize(labeled_frame, (540, 960))
        cv2.imshow('mask_ycrcb', cv2.bitwise_and(labeled_frame, labeled_frame, mask=mask))
        cv2.imshow('frame',labeled_frame)
        # exit on ESC press
        if cv2.waitKey(5) == 27:
            break
    
     
    
    print ("Is this weight ok?")
    if(check_weight):
            check_weight = False
            wave_obj = sa.WaveObject.from_wave_file("audio/weight.wav")
            play_obj = wave_obj.play()
    wait_for_thumbs_up(acf)
    
    ar_marker.reset()

# clean up
acf.release()
cv2.destroyAllWindows()
