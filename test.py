import cv2
import matplotlib.pyplot as plt
import skindetect
import gestures
from camfeed import AndroidCamFeed

#___________________________________
#host = "10.42.0.128:8080"

### Create new AndroidCamFeed instance
#acf = AndroidCamFeed(host)
#
### While camera is open
#while acf.isOpened():
#   ## Read frame
#   ret, frame = acf.read()
#   if ret:
#       skindetect.set_skin_threshold_from_face(frame)
#       mask = skindetect.process(frame)
#       
#       hand_contour = gestures.find_hand_contour(mask)
#       frame, hand_center,hand_radius = gestures.get_hand_center(frame,hand_contour)
#       frame,fingers,finger_count = gestures.mark_fingers(frame, hand_contour, hand_center, hand_radius)
#       frame=gestures.find_gesture(frame,finger_count)
#       cv2.imshow('feed', frame)
# 
#   if cv2.waitKey(1) == ord('q'):
#       break
#
### Must Release ACF instance
#acf.release()
#cv2.destroyAllWindows()

#______________________________________________________________
# setup capture
camera = cv2.VideoCapture(0)

# capture loop
while True:
    # get frame
    ret, frame = camera.read()    
         
    # mirror the frame (my camera mirrors by default)
    frame = cv2.flip(frame, 1)
    
    # store frame on pressing space
    wait = cv2.waitKey(5)
    if cv2.waitKey(5) == 27:
       break
    
    skindetect.set_skin_threshold_from_face(frame)
    mask = skindetect.process(frame)
       
    hand_contour = gestures.find_hand_contour(mask)
    frame, hand_center,hand_radius = gestures.get_hand_center(frame,hand_contour)
    frame,fingers,finger_count = gestures.mark_fingers(frame, hand_contour, hand_center, hand_radius)
    frame=gestures.find_gesture(frame,finger_count)
    cv2.imshow('feed', frame)
    
#______________________________________________________________
#frame = cv2.imread('test_data/gestures/five.jpg')
#skindetect.set_skin_threshold_from_face(frame)
#mask = skindetect.process(frame)
#   
#hand_contour = gestures.find_hand_contour(mask)
#frame, hand_center,hand_radius = gestures.get_hand_center(frame,hand_contour)
#frame,fingers,finger_count = gestures.mark_fingers(frame, hand_contour, hand_center, hand_radius)
#frame=gestures.find_gesture(frame,finger_count)
#
#plt.imshow(frame)
#cv2.imwrite('gesture.jpg', frame)
