import cv2
from camfeed import AndroidCamFeed

host = "192.168.2.14:8080"

## Create new AndroidCamFeed instance
acf = AndroidCamFeed(host)

## While camera is open
while acf.isOpened():
    ## Read frame
    ret, frame = acf.read()
    if ret:
        cv2.imshow('feed', frame)
    if cv2.waitKey(1) == ord('q'):
        break

## Must Release ACF instance
acf.release()
cv2.destroyAllWindows()