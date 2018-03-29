#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy
'''
REFERENCE: 

 
USAGE: 
img_col = cv2.imread(image_path, 1)
img_msk = skin_detector.process(img_col)
'''
class SkinDetect:
    # Lower and upper threshold for detecting skin YCrCb
    lower_threshold = [1,100,140]
    upper_threshold = [230,120,169]
    
    # Offset for YCrCb values
    yoffset = 100
    coffset = 15
    
    # Ranges of values of YCrCb between which skin color can be present
    LOWER_LIMIT = [1, 100, 130]
    UPPER_LIMIT = [230, 120, 180]
    
    # Returns rectangle coordinates for largest face in image
    def face_detect(self, img):
        haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_profileface.xml')

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_face_cascade.detectMultiScale(gray_img, 1.3, 5);
        
        area = 0
        largest_face = []
        if len(faces) == 0:
            return largest_face, False
        else:
            for (x, y, w, h) in faces:
                if (w*h > area):
                    largest_face = [x, y, w, h]
                    area = w*h
        return largest_face, True
    
    # Returns rectangle coordinates for left eye    
    def eye_detect(self, img):
        haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = haar_face_cascade.detectMultiScale(gray_img, 1.3, 5);
        
        rightmost_eye = []
        x_coords = 0
        if len(eyes) == 0:
            return [], False
        else:
            for (x, y, w, h) in eyes:
                if (x > x_coords):
                    rightmost_eye = [x, y, w, h]
                    x_coords = x
            return rightmost_eye, True
        
    # Set skin detection threshold from face
    def set_skin_threshold_from_face(self, img):        
        patch_ycrcb, face_coords, success_flag = self.get_patch_from_face(img)
        if (success_flag):            
            ycrcb_min, ycrcb_max = self.get_ycrcb_min_max(patch_ycrcb)

            self.lower_threshold = [max((ycrcb_min[0] - self.yoffset),self.LOWER_LIMIT[0]), 
                                    max((ycrcb_min[1] - self.coffset),self.LOWER_LIMIT[1]), 
                                    max((ycrcb_min[2] - self.coffset),self.LOWER_LIMIT[2])]
            self.upper_threshold = [min((ycrcb_max[0] + self.yoffset),self.UPPER_LIMIT[0]), 
                                    min((ycrcb_max[1] + self.coffset),self.UPPER_LIMIT[1]), 
                                    min((ycrcb_max[2] + self.coffset),self.UPPER_LIMIT[2])]          
            
            print (self.lower_threshold, " ", self.upper_threshold)
        return face_coords, success_flag
        
    # Gets patch of skin from under the eyes
    def get_patch_from_face(self, img):
        patch_size = 10
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        y, cr, cb = cv2.split(img_ycrcb)
        y = clahe.apply(y)
        img_enhanced_ycrcb = cv2.merge((y,cr,cb))
        
        # Detect face
        face, success_flag = self.face_detect(img)
        if (success_flag):
            x,y,w,h = face
        
            face_crop = img[y:y+h, x:x+w]
            face_enhanced_ycrcb = img_enhanced_ycrcb[y:y+h, x:x+w]
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            
            ref_x = x 
            ref_y = y
            #Detect eyes
            eyes, success_flag = self.eye_detect(face_crop)
            if (success_flag):
                x,y,w,h = eyes
                
                # Set offset to get patch of skin from size of eyes
                offset = int(h/4)
                
                # Get a patch below the eyes   
                x = x+int(3*w/4)
                y = y+h+offset
                
                ref_x += x 
                ref_y += y
                
                patch_ycrcb = face_enhanced_ycrcb[y:y+patch_size, x:x+patch_size]
                cv2.rectangle(img, (ref_x,ref_y), (ref_x+patch_size, ref_y+patch_size), (0,255,0), 2)
                return patch_ycrcb, face, True
            else:
                return [], [], False    
        else:
            return [], [], False
    
    # Gets the min and max YCrCb values from an image
    def get_ycrcb_min_max(self, img_ycrcb):        
        # Set min and max values of Y, Cr,Cb
        y, cr, cb = cv2.split(img_ycrcb)
        ycrcb_min = [numpy.amin(y), numpy.amin(cr), numpy.amin(cb)]
        ycrcb_max = [numpy.amax(y), numpy.amax(cr), numpy.amax(cb)]
        
        return ycrcb_min, ycrcb_max
        
    def get_ycrcb_mask(self, img):
        assert isinstance(img, numpy.ndarray), 'image must be a numpy array'
        assert img.ndim == 3, 'skin detection can only work on color images'
    
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        y, cr, cb = cv2.split(img_ycrcb)
        y = clahe.apply(y)
        img_ycrcb = cv2.merge((y,cr,cb))
        
        msk_ycrcb = cv2.inRange(img_ycrcb, numpy.array(self.lower_threshold), numpy.array(self.upper_threshold))
    
        return msk_ycrcb    
    
    def closing(self, mask):
        assert isinstance(mask, numpy.ndarray), 'mask must be a numpy array'
        assert mask.ndim == 2, 'mask must be a greyscale image'
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
        return mask
    
    
    def process(self, img):
        assert isinstance(img, numpy.ndarray), 'image must be a numpy array'
        assert img.ndim == 3, 'skin detection can only work on color images'
    
        mask = self.get_ycrcb_mask(img)
    
        mask = self.closing(mask)
    
        return mask