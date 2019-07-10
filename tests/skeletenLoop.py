import cv2
import numpy as np
from libs.skeleton import Skeleton
from libs.ar_marker import ARMarker
from libs.Point import point
import simpleaudio as sa

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 2
FONT_THICKNESS = 3
FONT_COLOR = (0,255,0)
FONT_COLOR_2 = (0,255,255)
TEXT_POSITION_X = 0.10
TEXT_POSITION_X_2 = 0.60
TEXT_POSITION_Y = 0.80

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

cap = cv2.VideoCapture('../test_data/markers/2.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('metric.avi',fourcc,30,(540, 960))

ar_marker = ARMarker()
isFirst = True
liftingStage = True
setUpStage = True
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if(not ret):
        continue
      
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

    
    mask = cv2.resize(mask, (540, 960))        
    labeled_frame = cv2.resize(labeled_frame, (540, 960))
    out.write(labeled_frame)
    cv2.imshow('mask_ycrcb', cv2.bitwise_and(labeled_frame, labeled_frame, mask=mask))
    cv2.imshow('frame',labeled_frame)
    # exit on ESC press
    if cv2.waitKey(5) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()