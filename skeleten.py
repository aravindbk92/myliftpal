from Point import point
import numpy as np
import cv2
import math
import simpleaudio as sa




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
        self.setUpkneeAngle = 88
        self.setupHipAngle = 110
        self.setupSholderAngle = 85
        self.triArea = 1700
        self.tracking = False
        self.liftingStage = 0
        self.barbell_count = 0
        self.barbell_start = False
        self.hip_count = 0
        self.hip_start = False
        self.knee_count = 0
        self.knee_start = False
        self.upperback_count = 0
        self.upperback_start = False
        self.lowerback_count = 0
        self.lowerback_start = False
    
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
        
    def hip_dif_angle(self, img):
        
        return True
    
    def setup_metrics(self, img,barbell):
        if(not self.tracking):
            return False
        else:
            if(self.liftingStage>45):
                return True
            a = self.hip_angle(img) 
            b = self.knee_angle(img) 
            c = self.upperback_pos(img) 
            d = self.lowerback_pos(img) 
            e = self.barbell_placement(img,barbell)
            if (a and b and c and d and e):
                self.liftingStage += 1
            return(self.liftingStage>45)
            
    def lifting_metrics(self, img,barbellPt):
        if(not self.tracking):
            return False
        else:
            a = self.barbell_placement(img,barbellPt)
            c = self.upperback_pos(img) 
            d = self.lowerback_pos(img) 
            return(c and d)

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
            elif(cy>self.hipPos.y+30 and cy<self.footPos.y-5):
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
            self.tracking = True            
            img, contour = self.tracker(img,contour)
            return img, contour
        self.tracking = False
        img, contour = self.detect_sholder(img, contour)
        img, contour = self.detect_hip(img, contour)
        img, contour = self.detect_spine(img, contour)
        img, contour = self.detect_foot(img, contour)
        img, contour = self.detect_knee(img, contour)
        
        return img, contour
    
    def tri_area(self, p1, p2, p3):
        area = ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2
        if (area > 0):
            return area
        else:
            return -area 
    
    def angle(self, p0, p1, p2):
        a = (p1.x-p0.x)**2 + (p1.y-p0.y)**2
        b = (p1.x-p2.x)**2 + (p1.y-p2.y)**2
        c = (p2.x-p0.x)**2 + (p2.y-p0.y)**2
        return math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi
        
    
    
    def barbell_placement(self,img,barbell):
        centerPnt = int((self.hipPos.x + self.footPos.x)/2)-75   
        if (centerPnt+100>barbell.x and centerPnt-100<barbell.x):
            cv2.circle(img, (barbell.x,barbell.y) , 25,(0,255,0),-1)
            self.barbell_count = 0
            self.barbell_start = True
            return True
        else:
            cv2.circle(img, (barbell.x,barbell.y) , 25,(0,0,255),-1)
            self.barbell_count += 1
            if(self.barbell_count > 40 and self.barbell_start):         
                wave_obj = sa.WaveObject.from_wave_file("audio/barbell.wav")
                play_obj = wave_obj.play()
                self.barbell_count = 0
            return False
            
        
    def knee_angle(self, img):
        kneeAngle = self.angle(self.hipPos,self.kneePos,self.footPos)
        if(kneeAngle < self.setUpkneeAngle+10 and kneeAngle > self.setUpkneeAngle-10):
            cv2.circle(img, (self.kneePos.x,self.kneePos.y) , 25,(0,255,0),-1)
            self.knee_count = 0
            self.knee_start = True
            return True
        else:
            cv2.circle(img, (self.kneePos.x,self.kneePos.y) , 25,(0,0,255),-1)
            self.knee_count += 1
            if(self.knee_count > 60 and self.knee_start):         
                wave_obj = sa.WaveObject.from_wave_file("audio/knees.wav")
                play_obj = wave_obj.play()
                self.knee_count = 0
            return False
        
    def hip_angle(self, img):
        self.spinePos[0].x = self.spinePos[0].x - 70
        hipAngle = self.angle(self.spinePos[0],self.hipPos,self.kneePos)
        self.spinePos[0].x = self.spinePos[0].x + 70
        #cv2.putText(img, str(hipAngle), (self.hipPos.x+100,self.hipPos.y+100) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),10)
        if(hipAngle < self.setupHipAngle+10 and hipAngle > self.setupHipAngle-10):
            cv2.circle(img, (self.hipPos.x,self.hipPos.y) , 25,(0,255,0),-1)
            self.hip_count = 0
            self.hip_start = True
            return True
        else:
            cv2.circle(img, (self.hipPos.x,self.hipPos.y) , 25,(0,0,255),-1)
            self.hip_count += 1
            if(self.hip_count > 60 and self.hip_start):         
                wave_obj = sa.WaveObject.from_wave_file("audio/hips.wav")
                play_obj = wave_obj.play()
                self.hip_count = 0
            return False
    
    def upperback_pos(self,img):
        sholderAngle = self.angle(self.sholderPos,self.spinePos[3],self.spinePos[2])
        if(sholderAngle < self.setupSholderAngle+10 and sholderAngle > self.setupSholderAngle-10):
            cv2.circle(img, (self.sholderPos.x,self.sholderPos.y) , 25,(0,255,0),-1)
            self.upperback_count = 0
            self.upperback_start = True
            return True
        else:
            cv2.circle(img, (self.sholderPos.x,self.sholderPos.y) , 25,(0,0,255),-1)
            self.upperback_count += 1
            if(self.upperback_count > 60 and self.upperback_start):         
                wave_obj = sa.WaveObject.from_wave_file("audio/sholder.wav")
                play_obj = wave_obj.play()
                self.upperback_count = 0
            return False
        
    def lowerback_pos(self,img):
        backPos = self.tri_area(self.spinePos[0],self.spinePos[1],self.spinePos[3])
        if(backPos < self.triArea):
            cv2.line(img,(self.spinePos[3].x,self.spinePos[3].y),(self.spinePos[0].x,self.spinePos[0].y),(0,255,0),15)
            self.lowerback_count = 0
            self.lowerback_start = True
            return True
        else:
            cv2.line(img,(self.spinePos[3].x,self.spinePos[3].y),(self.spinePos[0].x,self.spinePos[0].y),(0,0,255),15)
            self.lowerback_count += 1
            if(self.lowerback_count > 60 and self.lowerback_start):         
                wave_obj = sa.WaveObject.from_wave_file("audio/lowerBack.wav")
                play_obj = wave_obj.play()
                self.lowerback_count = 0
            return False
        
        
        
    
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