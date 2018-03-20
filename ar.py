# -*- coding: utf-8 -*-
import urllib
import cv2
import numpy as np
import time

url='http://10.20.42.202:8080/shot.jpg'

while True:

    # Use urllib to get the image and convert into a cv2 usable format
    imgResp=urllib.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)

    # put the image on screen
    cv2.imshow('IPWebcam',img)

    #To give the processor some less stress
    #time.sleep(0.1) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#import cv2
#import urllib2
#import numpy as np
#import sys
#
#host = "192.168.0.220:8080"
#if len(sys.argv)>1:
#    host = sys.argv[1]
#
#hoststr = 'http://' + host + '/video'
#print 'Streaming ' + hoststr
#
#stream=urllib2.urlopen(hoststr)
#
#bytes=''
#while True:
#    bytes+=stream.read(1024)
#    a = bytes.find('\xff\xd8')
#    b = bytes.find('\xff\xd9')
#    if a!=-1 and b!=-1:
#        jpg = bytes[a:b+2]
#        bytes= bytes[b+2:]
#        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
#        cv2.imshow(hoststr,i)
#        if cv2.waitKey(1) ==27:
#            exit(0)
    