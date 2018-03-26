import cv2
from skindetect import SkinDetect
from gestures import Gestures
from camfeed import AndroidCamFeed
import matplotlib.pyplot as plt

#______________________________________________________________
#setup capture
camera = cv2.VideoCapture(0)
skindetect = SkinDetect()
gestures = Gestures()

calibrate_timer = 0
calibrate_interval = 30
# capture loop
while True:
    # get frame
    ret, frame = camera.read()
         
    # mirror the frame (my camera mirrors by default)
    frame = cv2.flip(frame, 1)
    
    if (calibrate_timer % calibrate_interval == 0):
        if (not skindetect.set_skin_threshold_from_face(frame)):
            calibrate_timer-=1
 
    mask = skindetect.process(frame)
       
    if (calibrate_timer % calibrate_interval == 1):
        gestures.set_area_threshold(mask)
        calibrate_timer = 0
    calibrate_timer+=1
    
    cv2.imshow('mask', cv2.bitwise_and(frame, frame, mask=mask))
       
    frame, right_gesture, left_gesture = gestures.process(frame, mask)
       
    cv2.imshow('feed', frame)

    if cv2.waitKey(1) == ord('q'):
       break
    
# clean up
cv2.destroyAllWindows()
camera.release()
cv2.waitKey(1) # extra waitKey sometimes needed to close camera window