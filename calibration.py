# -*- coding: utf-8 -*-
import cv2
import glob
import pickle

'''
USAGE:
calibration = Calibration()
img = cv2.imread("ar.jpg", cv2.IMREAD_GRAYSCALE)
res = calibration.detect_marker(img)
cv2.aruco.drawDetectedMarkers(img,res[0],res[1])
plt.imshow(img)
'''
class Calibration:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(3,3,0.025,0.0125, dictionary)
    barbell_marker = cv2.aruco.drawMarker(dictionary, 13, 500)
    marker_length = 0.082
    pkl_file = 'calibration_matrix.pkl'
    
    def __init__(self):
        retval, self.cameraMatrix, self.distCoeffs, self.rvecs, self.tvecs = self.get_saved_calibration_matrix()        
        
    # dumps calibration matrix to pickle file
    def calc_and_save_calibration_matrix(self, boards_path='/boards', out_pkl_file=pkl_file):
        if (out_pkl_file != self.pkl_file):
            self.pkl_file = out_pkl_file
        images = glob.glob(boards_path + '/*.jpg')
 
        allCorners = []
        allIds = []
        decimator = 0
        for fname in images:
            frame = cv2.imread(fname)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            res = cv2.aruco.detectMarkers(gray, self.dictionary)
            
            if len(res[0])>0:
                res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,self.board)
                if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%3==0:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])
        
                cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
            decimator+=1
            
        imsize = gray.shape   
        try:
            cal = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,self.board,imsize,None,None)
            print (cal)
            print ("Calibration matrix saved to " + out_pkl_file)
            with open(out_pkl_file, 'wb') as f:
                pickle.dump(cal, f)
        except:
            print("Calibration failed. Needs >10 board images from different angles.")

    # returns board image
    def get_calibration_board_image(self):
        img = self.boardboard.draw((200*3,200*3))
        return img
        
    # returns (retval, cameraMatrix, distCoeffs, rvecs, tvecs)
    def get_saved_calibration_matrix(self, in_pkl_file=pkl_file):
        print ("Reading calibation matrix from " + in_pkl_file)
        cal = None
        
        try:
            with open(in_pkl_file, 'rb') as f:
                cal = pickle.load(f)
        except:
            print ("Could not find saved calibration matrix.")
        
        return cal
    
    # returns (corners, ids, rejectedImgPoints)
    def detect_marker(self, frame):
        res = cv2.aruco.detectMarkers(frame, self.dictionary)
        
        if len(res[0]) > 0:
            return res
    # returns (rvecs, tvecs, _objPoints)
    def get_marker_pose(self, frame):
        res = self.detect_marker(frame)
        return cv2.aruco.estimatePoseSingleMarkers(res[0], self.marker_length, self.cameraMatrix, self.distCoeffs)