import cv2
import numpy as np
from skeleten import Skeleton

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
    
skeleton = Skeleton()

cap = cv2.VideoCapture('test_data/markers/2.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    
    mask = apply_mask(frame)    
    
    #find all contours in the screen
    im2,contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    labeled_frame , contourskeleton = skeleton.delect_skeleton(frame, contours)

    labeled_frame = skeleton.draw_skeleton(labeled_frame)    
    
    #draw all contours in the the frame so they are visible
    cv2.drawContours(labeled_frame, contours, -1, (255,255,255), 1)
    
    mask = cv2.resize(mask, (540, 700))        
    labeled_frame = cv2.resize(labeled_frame, (540, 700))
    cv2.imshow('mask_ycrcb', cv2.bitwise_and(labeled_frame, labeled_frame, mask=mask))
    cv2.imshow('frame',labeled_frame)
    # exit on ESC press
    if cv2.waitKey(5) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()