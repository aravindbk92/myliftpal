#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
'''
REFERENCE: 
https://github.com/mahaveerverma/hand-gesture-recognition-opencv

USAGE: 
hand_contour = gestures.find_hand_contour(mask)
frame, hand_center,hand_radius = gestures.get_hand_center(frame,hand_contour)

frame,fingers,finger_count = gestures.mark_fingers(frame, hand_contour, hand_center, hand_radius)
frame=gestures.find_gesture(img,finger_count)
cv2.imshow('feed', frame)
'''

first_iteration=True
finger_ct_history=[0,0]
finger_thresh_l=2.0
finger_thresh_u=3.8

# Returns the contour of the leftmost blob (> an area threshold)
# This corresponds to the right hand in the skin detection mask
def find_hand_contour(mask):
    area_threshold = 5000    
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    leftmost_blob_index = 0
    leftmost_x = 10000
    for index, cnt in enumerate(contours):
        # Find area of contour
        area = cv2.contourArea(cnt)
        
        # Calculate centre of mass from moments
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        
        if area > area_threshold and cx < leftmost_x:
            leftmost_blob_index = index
            leftmost_x = cx
           
    return contours[leftmost_blob_index]

# Finds the center of the largest circle inscribed inside the contour
# This corresponds to the centre of the palm.
# Returns center of this circle and its radius    
def get_hand_center(img, contour):
    max_d=0
    pt=(0,0)
    x,y,w,h = cv2.boundingRect(contour)
    for ind_y in range(int(y+0.3*h),int(y+0.8*h)): #around 0.25 to 0.6 region of height (Faster calculation with ok results)
        for ind_x in range(int(x+0.3*w),int(x+0.6*w)): #around 0.3 to 0.6 region of width (Faster calculation with ok results)
            dist= cv2.pointPolygonTest(contour,(ind_x,ind_y),True)
            if(dist>max_d):
                max_d=dist
                pt=(ind_x,ind_y)
    
    cv2.circle(img, pt, int(max_d), (255,0,0))
    
    return img, pt, max_d

# Using the center of palm as reference, eliminate all points 
# from the convex hull which do not seem to be part of hand.
def mark_fingers(img,contour,center,radius):
    global first_iteration
    global finger_ct_history
    
    hull = cv2.convexHull(contour)
    
    finger=[(hull[0][0][0],hull[0][0][1])]
    j=0

    cx = center[0]
    cy = center[1]
    
    for i in range(len(hull)):
        dist = np.sqrt((hull[-i][0][0] - hull[-i+1][0][0])**2 + (hull[-i][0][1] - hull[-i+1][0][1])**2)
        if (dist>18):
            if(j==0):
                finger=[(hull[-i][0][0],hull[-i][0][1])]
            else:
                finger.append((hull[-i][0][0],hull[-i][0][1]))
            j=j+1
    
    temp_len=len(finger)
    i=0
    while(i<temp_len):
        dist = np.sqrt( (finger[i][0]- cx)**2 + (finger[i][1] - cy)**2)
        if(dist<finger_thresh_l*radius or dist>finger_thresh_u*radius or finger[i][1]>cy+radius):
            finger.remove((finger[i][0],finger[i][1]))
            temp_len=temp_len-1
        else:
            i=i+1        
    
    temp_len=len(finger)
    if(temp_len>5):
        for i in range(1,temp_len+1-5):
            finger.remove((finger[temp_len-i][0],finger[temp_len-i][1]))
    
    if(first_iteration):
        finger_ct_history[0]=finger_ct_history[1]=len(finger)
        first_iteration=False
    else:
        finger_ct_history[0]=0.34*(finger_ct_history[0]+finger_ct_history[1]+len(finger))

    if((finger_ct_history[0]-int(finger_ct_history[0]))>0.8):
        finger_count=int(finger_ct_history[0])+1
    else:
        finger_count=int(finger_ct_history[0])

    finger_ct_history[1]=len(finger)

    for k in range(len(finger)):
        cv2.circle(img,finger[k],10,255,2)
        cv2.line(img,finger[k],(cx,cy),255,2)
    
    return img,finger, finger_count
    
def find_gesture(img,finger_count):
    count_gestures = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
    gesture_text="GESTURE: "+count_gestures[finger_count]
    cv2.putText(img,gesture_text,(int(0.56*img.shape[1]),int(0.97*img.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
    return img
    