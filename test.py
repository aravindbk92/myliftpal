import cv2
from skindetect import SkinDetect
from gestures import Gestures
from camfeed import AndroidCamFeed
import matplotlib.pyplot as plt

#___________________________________
#host = "10.42.0.128:8080"
#
### Create new AndroidCamFeed instance
#acf = AndroidCamFeed(host)
#skindetect = SkinDetect()
#gestures = Gestures()
#
### While camera is open
#while acf.isOpened():
#   ## Read frame
#   ret, frame = acf.read()
#   if ret:
#       skindetect.set_skin_threshold_from_face(frame)
#       mask = skindetect.process(frame)
#       
#       cv2.imshow('mask', cv2.bitwise_and(frame, frame, mask=mask))
#       
#       frame, gesture= gestures.process(frame, mask)
#       
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
#camera = cv2.VideoCapture(0)
#skindetect = SkinDetect()
#gestures = Gestures()
#
## capture loop
#while True:
#    # get frame
#    ret, frame = camera.read()    
#         
#    # mirror the frame (my camera mirrors by default)
#    frame = cv2.flip(frame, 1)
#    
#    skindetect.set_skin_threshold_from_face(frame)
#    mask = skindetect.process(frame)
#    
#    cv2.imshow('mask', cv2.bitwise_and(frame, frame, mask=mask))
#       
#    frame, gesture= gestures.process(frame, mask)
#       
#    cv2.imshow('feed', frame)
#
#    if cv2.waitKey(1) == ord('q'):
#       break
#    
## clean up
#cv2.destroyAllWindows()
#camera.release()
#cv2.waitKey(1) # extra waitKey sometimes needed to close camera window
    
#______________________________________________________________
#frame = cv2.imread('test_data/gestures/five.jpg')
#
#skindetect = SkinDetect()
#gestures = Gestures()
#skindetect.set_skin_threshold_from_face(frame)
#mask = skindetect.process(frame)
#
#frame, gesture = gestures.process(frame, mask)
#
#plt.imshow(frame)
#cv2.imwrite('gesture.jpg', frame)
