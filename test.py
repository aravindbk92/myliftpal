import cv2
import matplotlib.pyplot as plt
import skindetect
import gestures

 host = "192.168.2.14:8080"

 ## Create new AndroidCamFeed instance
 acf = AndroidCamFeed(host)

 ## While camera is open
 while acf.isOpened():
    ## Read frame
    ret, frame = acf.read()
    if ret:
        skindetect.set_skin_threshold_from_face(frame)
        mask = skindetect.process(frame)
        
        hand_contour = gestures.find_hand_contour(mask)
        frame, hand_center,hand_radius = gestures.get_hand_center(frame,hand_contour)

        frame,fingers,finger_count = gestures.mark_fingers(frame, hand_contour, hand_center, hand_radius)
        frame=gestures.find_gesture(img,finger_count)
        cv2.imshow('feed', frame)
    if cv2.waitKey(1) == ord('q'):
        break

 ## Must Release ACF instance
 acf.release()
 cv2.destroyAllWindows()
