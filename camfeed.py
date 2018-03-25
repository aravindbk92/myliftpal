#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import threading
from urllib.request import urlopen
import urllib
from socket import error as SocketError

'''
USAGE
## Set host
host = 192.168.1.2:8080

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
'''
class AndroidCamFeed:
    __bytes=b''
    __stream = None
    __isOpen = False
    __feed = None
    __noStreamCount = 1
    __loadCode = cv2.IMREAD_COLOR if sys.version_info[0] > 2 \
                                    else cv2.CV_LOAD_IMAGE_COLOR
    def __init__(self, host):
        self.hoststr = 'http://' + host + '/video'
        try:
            AndroidCamFeed.__stream = urlopen(self.hoststr, timeout = 3)
            AndroidCamFeed.__isOpen = True
        except (SocketError, urllib.URLError) as err:
            print ("Failed to connect to stream. \nError: " + str(err))
            self.__close()
        t = threading.Thread(target=self.__captureFeed)
        t.start()

    def __captureFeed(self):
        while AndroidCamFeed.__isOpen:
            newbytes = AndroidCamFeed.__stream.read(1024)
            if not newbytes:
                self.__noStream()
                continue
            AndroidCamFeed.__bytes += newbytes
            self.a = AndroidCamFeed.__bytes.find(b'\xff\xd8')
            self.b = AndroidCamFeed.__bytes.find(b'\xff\xd9')
            if self.a != -1 and self.b != -1:
                self.jpg = AndroidCamFeed.__bytes[self.a : self.b+2]
                AndroidCamFeed.__bytes = AndroidCamFeed.__bytes[self.b+2 : ]
                AndroidCamFeed.__feed = cv2.imdecode(np.fromstring(self.jpg,
                                                dtype = np.uint8),
                                                    AndroidCamFeed.__loadCode)
        return

    def __close(self):
        AndroidCamFeed.__isOpen = False
        AndroidCamFeed.__noStreamCount = 1

    def __noStream(self):
        AndroidCamFeed.__noStreamCount += 1
        if AndroidCamFeed.__noStreamCount > 10:
            try:
                AndroidCamFeed.__stream = urlopen(
                                            self.hoststr, timeout = 3)
            except (SocketError, urllib.URLError) as err:
                print ("Failed to connect to stream: Error: " + str(err))
                self.__close()

    def isOpened(self):
        return AndroidCamFeed.__isOpen

    def read(self):
        if AndroidCamFeed.__feed is not None:
            return True, AndroidCamFeed.__feed
        else:
            return False, None

    def release(self):
        self.__close()
