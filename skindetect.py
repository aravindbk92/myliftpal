#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import cv2
import numpy
'''
REFERENCE: 
https://github.com/WillBrennan/SkinDetector/tree/master/skin_detector
 
USAGE: 
img_col = cv2.imread(image_path, 1)
img_msk = skin_detector.process(img_col)
'''

logger = logging.getLogger('main')
ycrcb_min = [122,158,103]
ycrcb_max = [134,158,103]

# Returns rectangle coordinates for largest face in image
def face_detect(img):
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
def eye_detect(img):
    haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = haar_face_cascade.detectMultiScale(gray_img, 1.3, 5);
    
    if (len(eyes) == 2):
        return eyes[0], True
    else:
        return [], False

# Set skin detection threshold from face
def set_skin_threshold_from_face(img):
    global ycrcb_min, ycrcb_max
    patch, success_flag = get_patch_from_face(img)
    if (success_flag):
        ycrcb_min, ycrcb_max = get_ycrcb_min_max(patch)
    
# Gets patch of skin from under the eyes
def get_patch_from_face(img):
    patch_size = 10
    
    # Detect face
    face, success_flag = face_detect(img)
    if (success_flag):
        x,y,w,h = face
    
        face = img[y:y+h, x:x+w]
        
        #Detect eyes
        eyes, success_flag = eye_detect(face)
        if (success_flag):
            x,y,w,h = eyes
            
            # Set offset to get patch of skin from size of eyes
            offset = int(3*h/4)
            
            # Get a patch below the eyes   
            x = x+int(3*w/4)
            y = y+int(h/2)+offset
        
            patch = face[y:y+patch_size, x:x+patch_size]
            return patch, True
        else:
            return [], False    
    else:
        return [], False

# Gets the min and max YCrCb values from an image
def get_ycrcb_min_max(img):
    # Convert patch to yCrCb
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    # Set min and max values of Y, Cr,Cb
    y, cr, cb = cv2.split(img_ycrcb)
    ycrcb_min = [numpy.amin(y), numpy.amin(cr), numpy.amin(cb)]
    ycrcb_max = [numpy.amax(y), numpy.amax(cr), numpy.amax(cb)]
    
    return ycrcb_min, ycrcb_max
    
def get_ycrcb_mask(img):
    assert isinstance(img, numpy.ndarray), 'image must be a numpy array'
    assert img.ndim == 3, 'skin detection can only work on color images'    

    global ycrcb_min, ycrcb_max
    
    # Offset for YCrCb values
    yoffset = 50
    coffset = 11
    
    lower_thresh = numpy.array([ycrcb_min[0]-yoffset, ycrcb_min[1]-coffset, ycrcb_min[2]-coffset], dtype=numpy.uint8)
    upper_thresh = numpy.array([ycrcb_max[0]+yoffset, ycrcb_max[1]+coffset, ycrcb_max[2]+coffset], dtype=numpy.uint8)

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    msk_ycrcb = cv2.inRange(img_ycrcb, lower_thresh, upper_thresh)

    msk_ycrcb[msk_ycrcb < 128] = 0
    msk_ycrcb[msk_ycrcb >= 128] = 1

    return msk_ycrcb.astype(float)

def grab_cut_mask(img_col, mask):
    assert isinstance(img_col, numpy.ndarray), 'image must be a numpy array'
    assert isinstance(mask, numpy.ndarray), 'mask must be a numpy array'
    assert img_col.ndim == 3, 'skin detection can only work on color images'
    assert mask.ndim == 2, 'mask must be 2D'

    kernel = numpy.ones((50, 50), numpy.float32) / (50 * 50)
    dst = cv2.filter2D(mask, -1, kernel)
    dst[dst != 0] = 255
    free = numpy.array(cv2.bitwise_not(dst), dtype=numpy.uint8)

    grab_mask = numpy.zeros(mask.shape, dtype=numpy.uint8)
    grab_mask[:, :] = 2
    grab_mask[mask == 255] = 1
    grab_mask[free == 255] = 0

    if numpy.unique(grab_mask).tolist() == [0, 1]:
        bgdModel = numpy.zeros((1, 65), numpy.float64)
        fgdModel = numpy.zeros((1, 65), numpy.float64)

        if img_col.size != 0:
            mask, bgdModel, fgdModel = cv2.grabCut(img_col, grab_mask, None, bgdModel, fgdModel, 5,
                                                   cv2.GC_INIT_WITH_MASK)
            mask = numpy.where((mask == 2) | (mask == 0), 0, 1).astype(numpy.uint8)
        else:
            logger.warning('img_col is empty')

    return mask


def closing(mask):
    assert isinstance(mask, numpy.ndarray), 'mask must be a numpy array'
    assert mask.ndim == 2, 'mask must be a greyscale image'

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


def process(img):
    assert isinstance(img, numpy.ndarray), 'image must be a numpy array'
    assert img.ndim == 3, 'skin detection can only work on color images'

    mask = get_ycrcb_mask(img)

    mask = mask.astype(numpy.uint8)

    mask = closing(mask)
    mask = grab_cut_mask(img, mask)

    return mask