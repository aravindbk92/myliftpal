from Point import point
import numpy as np
import cv2

class Skeleton:
    
    spinePos = [];
    
    def __init__(self):
        for i in range(4):
            self.spinePos.append(point())
        self.sholderPos = point()
        self.hipPos = point()
        self.kneePos = point()
        self.footPos = point()
        self.startTracking = False
        self.offSetDisplay =40
        self.minDistance = 37
        self.startingLocation = point(5000,5000)        
        
    
    
    def detect_sholder(self, img, contour):
        cnt = contour
        if(cv2.contourArea(cnt[len(cnt)-1])<(cv2.contourArea(cnt[len(cnt)-2]))):
           center = cv2.moments(cnt[len(cnt)-2])
           cnt.remove(cnt[len(cnt)-2])
        else:
           center = cv2.moments(cnt[len(cnt)-1])
           cnt.remove(cnt[len(cnt)-1])
        if center['m00'] == 0:
            center['m00'] = 1
            # get mid-point
        self.sholderPos.x = int(center['m10']/center['m00'])
        self.sholderPos.y = int(center['m01']/center['m00'])
        cv2.putText(img, 'shoulder', (self.sholderPos.x,self.sholderPos.y) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),10)
        if(self.sholderPos.y<self.startingLocation.y):
            self.startingLocation.y = self.sholderPos.y
            self.startingLocation.x = self.sholderPos.x
        return img, cnt
        
    def detect_hip(self, img, contour):
        for cnt in contour:    
            center = cv2.moments(cnt)
            # get mid-point
            cx = int(center['m10']/center['m00'])
            cy = int(center['m01']/center['m00'])
            
            if(abs(self.sholderPos.x-cx)<20 or (self.sholderPos.x-cx>0 and self.sholderPos.x-cx < 160)):
                if(abs(self.hipPos.y-cy)<40):
                    cx = int((cx+self.hipPos.x)/2)
                    cy = int((cy+self.hipPos.y)/2)
                self.hipPos.x = cx
                self.hipPos.y = cy
        cv2.putText(img, 'hip', (self.hipPos.x,self.hipPos.y) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),10)
        return img, contour
    
    def detect_spine(self, img, contour):
        count = 0
        for cnt in contour:    
            center = cv2.moments(cnt)
            # get mid-point
            cx = int(center['m10']/center['m00'])
            cy = int(center['m01']/center['m00'])
            if(cy<self.hipPos.y-8 and cx>self.sholderPos.x+18):
                self.spinePos[count].x = cx
                self.spinePos[count].y = cy
                count = count + 1
            if (count == 4):
                break
        counter = 0
        for i in range(count):
            cv2.putText(img, str(counter), (self.spinePos[counter].x,self.spinePos[counter].y) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),10)
            counter = counter +1
        return img, contour
        
    def detect_knee(self, img, contour):
        for cnt in contour:    
            center = cv2.moments(cnt)
            # get mid-point
            cx = int(center['m10']/center['m00'])
            cy = int(center['m01']/center['m00'])
            
            if(abs(self.sholderPos.x-cx)<300 and self.hipPos.y+200<cy and self.footPos.y>cy ):
                self.kneePos.x = cx
                self.kneePos.y = cy
                cv2.putText(img, 'knee', (self.kneePos.x,self.kneePos.y) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),10)
        return img, contour
        
    def detect_foot(self, img, contour):
        cnt = contour[0]    
        center = cv2.moments(cnt)
            # get mid-point
        cx = int(center['m10']/center['m00'])
        cy = int(center['m01']/center['m00'])
            
        if(abs(self.sholderPos.x-cx)<300):
            self.footPos.x = cx
            self.footPos.y = cy
            cv2.putText(img, 'knee', (self.footPos.x,self.footPos.y) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),10)
        return img, contour
    
    def tracker(self,img,countour):
        for cnt in countour:
            center = cv2.moments(cnt)
            if center['m00'] == 0:
                center['m00'] = 1
            cx = int(center['m10']/center['m00'])
            cy = int(center['m01']/center['m00'])
            current = point(cx,cy)
            if(current.distance(self.sholderPos)<self.minDistance):
                self.sholderPos = current
                continue
            if(current.distance(self.hipPos)<self.minDistance):
                self.hipPos = current
                continue
            if(current.distance(self.kneePos)<self.minDistance):
                self.kneePos = current
                continue
            elif(cy>self.hipPos.y+10 and cy<self.footPos.y-5):
                self.kneePos = current
                continue
            if(current.distance(self.footPos)<self.minDistance):
                self.footPos = current
                continue
            for i in range(4):
                if(current.distance(self.spinePos[i])<self.minDistance):
                    self.spinePos[i] = current
                    continue    
        return img, countour
    
    def delect_skeleton(self, img, contour):
        cnt = 0
        for ball in self.spinePos:
            if(ball.x != 0 and ball.y != 0):
                cnt += 1
        if(self.kneePos.x != 0 and self.kneePos.y != 0 and self.hipPos.x != 0 and self.hipPos.y != 0 and self.sholderPos.x != 0 and self.sholderPos.y != 0 and cnt == 4 and self.sholderPos.y-self.startingLocation.y>10):
            img, contour = self.tracker(img,contour)
            return img, contour
        img, contour = self.detect_sholder(img, contour)
        img, contour = self.detect_hip(img, contour)
        img, contour = self.detect_spine(img, contour)
        img, contour = self.detect_foot(img, contour)
        img, contour = self.detect_knee(img, contour)
        
        return img, contour
        
    def draw_skeleton(self, img):
        cnt = 0
        last = point()
        for ball in self.spinePos:
            if(ball.x != 0 and ball.y != 0):
                if(cnt != 0):
                    cv2.line(img,(last.x,last.y),(ball.x,ball.y),(255,0,0),5)
                last.x = ball.x
                last.y = ball.y
                cnt += 1
        if(self.sholderPos.x != 0 and self.sholderPos.y != 0):                
            cv2.line(img,(self.sholderPos.x,self.sholderPos.y),(last.x ,last.y),(255,0,0),5)
        if(self.hipPos.x != 0 and self.hipPos.y != 0):
            cv2.line(img,(self.hipPos.x,self.hipPos.y),(self.spinePos[0].x,self.spinePos[0].y),(255,0,0),5)
        if(self.kneePos.x != 0 and self.kneePos.y != 0):
            cv2.line(img,(self.hipPos.x,self.hipPos.y),(self.kneePos.x ,self.kneePos.y),(255,0,0),5)
        if(self.footPos.x != 0 and self.footPos.y != 0):
            cv2.line(img,(self.footPos.x,self.footPos.y),(self.kneePos.x ,self.kneePos.y),(255,0,0),5)
        
        return img