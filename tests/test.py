#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 03:18:06 2018

@author: abk
"""

import cv2
from libs.ar_marker import ARMarker

VIEW_RESIZE = (540,960)
VIEW_WINDOW = 'frame'

ar_marker = ARMarker()
vidcap = cv2.VideoCapture('../test_data/office_deadlift.avi')
first = True
while (vidcap.isOpened()):
    ## Read frame
    vid_read_success,frame = vidcap.read()
    if (first):
        barbell_position, exercise = ar_marker.find_initial_barbell_position(frame)
        ar_marker.set_deadlift_rep_threshold(1150)
        print("frame_y: ", frame.shape[0])
        print("exercise: ", exercise)
        first = False
    else:
        barbell_position =  ar_marker.get_marker_center(frame)
        print ("barbell: ", barbell_position)
        cv2.circle(frame,barbell_position,10,(255,255,255),2)
        

        ar_marker.track_marker(barbell_position)
        for pos in ar_marker.marker_history:
            cv2.circle(frame,pos,5,(0,255,0),3)
        current_rep = ar_marker.count_reps(barbell_position)
        print ("current_rep: ", current_rep  , "\n")
    
    frame = cv2.resize(frame, VIEW_RESIZE)
    cv2.imshow(VIEW_WINDOW, frame)
            
    if cv2.waitKey(1) == ord('q'):
        break
    
vidcap.release()
cv2.destroyAllWindows()