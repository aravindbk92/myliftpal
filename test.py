import cv2
import matplotlib.pyplot as plt
import skindetect

# host = "192.168.2.14:8080"

# ## Create new AndroidCamFeed instance
# acf = AndroidCamFeed(host)

# ## While camera is open
# while acf.isOpened():
#     ## Read frame
#     ret, frame = acf.read()
#     if ret:
#         cv2.imshow('feed', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# ## Must Release ACF instance
# acf.release()
# cv2.destroyAllWindows()

img = cv2.imread("test_data/gestures/five_1.jpg")
mask = skindetect.process(img)
cv2.imwrite("result.png", cv2.bitwise_and(img, img, mask=mask))
plt.imshow(mask)