#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import traceback
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

class Gestures:
    first_iteration=True
    finger_ct_history=[0,0]
    finger_thresh_l=2.0
    finger_thresh_u=3.8
    area_threshold = 5000
    face_x = 0
    face_y = 0
    face_h = 10000
    
    # Sets area threshold for hands from information in mask
    def set_area_threshold(self, face_coords):
        self.area_threshold = int((face_coords[2] * face_coords[3])/16);

    
    # Returns the contour of the leftmost and rightmost blob (> an area threshold)
    # This corresponds to the right hand and left hand respectively
    def find_hand_contour(self, frame, mask, face_coords):
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (face_coords):
            self.face_x = face_coords[0]
            self.face_y = face_coords[1]
            self.face_h = face_coords[3]
        
        leftmost_blob_index = -1
        leftmost_x = 10000
        distance_below_face = 10000
        for index, cnt in enumerate(contours):
            # Find area of contour
            area = cv2.contourArea(cnt)
            
            # Calculate centre of mass from moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            else:
                cx = 10000
                cy = 0
            
            if ( area > self.area_threshold and cx < leftmost_x and cx < self.face_x and cy < self.face_y+2.5*self.face_h and cy > self.face_y-self.face_h):
                if cy < distance_below_face:
                    leftmost_blob_index = index
                    leftmost_x = cx
                    distance_below_face = cy
        
        hand = None
        if leftmost_blob_index >= 0 and leftmost_blob_index < len(contours):
            hand = contours[leftmost_blob_index]
            cv2.drawContours(frame, [hand], 0, (0,255,0), 3)
        
        return frame, hand
    
    # Finds the center of the largest circle inscribed inside the contour
    # This corresponds to the centre of the palm.
    # Returns center of this circle and its radius    
    def get_hand_center(self, img, contour):
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
    def mark_fingers(self, img,contour,center,radius):       
        hull = cv2.convexHull(contour)
        
        finger=[(hull[0][0][0],hull[0][0][1])]
        j=0
    
        cx = center[0]
        cy = center[1]
        
        if (len(hull)  > 1):
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
                if(dist<self.finger_thresh_l*radius or dist>self.finger_thresh_u*radius or finger[i][1]>cy+radius):
                    finger.remove((finger[i][0],finger[i][1]))
                    temp_len=temp_len-1
                else:
                    i=i+1        
            
            temp_len=len(finger)
            if(temp_len>5):
                for i in range(1,temp_len+1-5):
                    finger.remove((finger[temp_len-i][0],finger[temp_len-i][1]))
            
            if(self.first_iteration):
                self.finger_ct_history[0]=self.finger_ct_history[1]=len(finger)
                self.first_iteration=False
            else:
                self.finger_ct_history[0]=0.34*(self.finger_ct_history[0]+self.finger_ct_history[1]+len(finger))
        
            if((self.finger_ct_history[0]-int(self.finger_ct_history[0]))>0.8):
                finger_count=int(self.finger_ct_history[0])+1
            else:
                finger_count=int(self.finger_ct_history[0])
        
            self.finger_ct_history[1]=len(finger)
        
            for k in range(len(finger)):
                cv2.circle(img,finger[k],10,255,2)
                cv2.line(img,finger[k],(cx,cy),255,2)
            
            return img, finger, finger_count
        
        else:
            return img, [], 0 
        
    def find_gesture(self, img, finger_count):
        gestures = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE']
        gesture_text="GESTURE: "+gestures[finger_count]
        cv2.putText(img,gesture_text,(int(0.56*img.shape[1]),int(0.97*img.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
        return img, gestures[finger_count]
    
    def process(self, frame, mask, face_coords):
        try:
            frame, hand_contour = self.find_hand_contour(frame, mask, face_coords)
        
            gesture = None
            if hand_contour is not None:
                frame, hand_center, hand_radius = self.get_hand_center(frame, hand_contour)
                frame, fingers, finger_count = self.mark_fingers(frame, hand_contour, hand_center, hand_radius)
                frame, right_gesture = self.find_gesture(frame,finger_count)
        
        except Exception as e:
            print (e)
            traceback.print_exc()
            return frame, ""
        
        return frame, gesture