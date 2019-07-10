#!/uslibsr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy
'''
REFERENCE: 

 
USAGE: 
img_col = cv2.imread(image_path, 1)
img_msk = skin_detector.process(img_col)
'''
class SkeletonDetect:
    # Lower and upper threshold for detecting skin YCrCb
    lower_threshold = [10, 40, 120]
    upper_threshold = [225, 80, 145]
    
             
    def get_ycrcb_mask(self, img):
        assert isinstance(img, numpy.ndarray), 'image must be a numpy array'
        assert img.ndim == 3, 'skin detection can only work on color images'
    
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        
       
        msk_ycrcb = cv2.inRange(img_ycrcb, numpy.array(self.lower_threshold), numpy.array(self.upper_threshold))
    
        return msk_ycrcb    
    
    def closing(self, mask):
        assert isinstance(mask, numpy.ndarray), 'mask must be a numpy array'
        assert mask.ndim == 2, 'mask must be a greyscale image'
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
        return mask
    
    
    def process(self, img):
        assert isinstance(img, numpy.ndarray), 'image must be a numpy array'
        assert img.ndim == 3, 'skin detection can only work on color images'
    
        mask = self.get_ycrcb_mask(img)
    
        mask = self.closing(mask)
    
        return mask