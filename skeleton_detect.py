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
class SkeletonDetect:
    # Lower and upper threshold for detecting skin YCrCb
    lower_threshold = [10,100,140]
    upper_threshold = [230,120,169]
    
    # Offset for YCrCb values
    yoffset_l = 50
    yoffset_u = 50
    croffset_l = 20
    croffset_u = 2
    cyoffset_l = 2
    cyoffset_u = 20

    
    # Ranges of values of YCrCb between which skin color can be present
    LOWER_LIMIT = [55, 75, 130]
    UPPER_LIMIT = [230, 122, 180]
       
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