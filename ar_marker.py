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
class ARMarker:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(3,3,0.025,0.0125, dictionary)
    barbell_marker = cv2.aruco.drawMarker(dictionary, 13, 500)
    marker_length = 0.082
    pkl_file = 'data/calibration/calibration_matrix.pkl'
    
    exercise_type = "deadlift" #defualt
    reps = 0
    sets = 0
    current_rep = 0
    
    initial_barbell_position = None
    deadlift_rep_threshold = None
    squat_rep_threshold = None
    rep_count_flag = False
    
    marker_history = []
    history_length = 100
    
    def __init__(self):
        retval, self.cameraMatrix, self.distCoeffs, self.rvecs, self.tvecs = self.get_saved_calibration_matrix()        
        
    # dumps calibration matrix to pickle file
    def calc_and_save_calibration_matrix(self, boards_path='data/calibration/boards', out_pkl_file=pkl_file):
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
        
    # returns center of single aruco marker
    def get_marker_center(self, frame):
        corners, id, rejectedPts =  self.detect_marker(frame)
        
        p1, p2, p3, p4 = corners[0][0]
        L1 = self.line(p1, p3)
        L2 = self.line(p2, p4)
        
        return self.intersection(L1, L2)
        
    def line(self, p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C
    
    def intersection(self, L1, L2):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x,y
        else:
            return False
        
    # returns (rvecs, tvecs, _objPoints)
    def get_marker_pose(self, frame):
        res = self.detect_marker(frame)
        return cv2.aruco.estimatePoseSingleMarkers(res[0], self.marker_length, self.cameraMatrix, self.distCoeffs)
    
    def find_initial_barbell_position(self, frame):
        center =  self.get_marker_center(frame)
        if center is not False:
            self.identify_exercise(frame, center)
            self.initial_barbell_position = center
            return self.initial_barbell_position, self.exercise_type
        else:
            return False
                
    def identify_exercise(self, frame, barbell_position):
        if barbell_position[1] < frame.shape[0]/2:
            self.exercise_type = "squat"
        else:
            self.exercise_type = "deadlift"
            
    def count_reps(self, barbell_position):        
        if (self.exercise_type == 'deadlift'):
            if barbell_position[1] < self.deadlift_rep_threshold:
                if self.rep_count_flag == False:
                    self.current_rep+=1
                    self.rep_count_flag=True
            else:
                self.rep_count_flag=False
        elif (self.exercise_type == 'squat'):
            if barbell_position[1] < self.squat_rep_threshold:
                if self.rep_count_flag == False:
                    self.current_rep+=1
                    self.rep_count_flag=True
            else:
                self.rep_count_flag=False
                
        return self.current_rep
                
    def reset(self):        
        self.exercise_type = "deadlift" #default
        self.reps = 0
        self.sets = 0
        self.current_rep = 0
        
        self.initial_barbell_position = None
        self.barbell_position = None
        self.deadlift_rep_threshold = None
        self.squat_rep_threshold = None
        self.rep_count_flag = False
        
        self.track_marker_array = []
        
    def set_deadlift_rep_threshold(self, threshold_y):        
        self.deadlift_rep_threshold = threshold_y
        
    def set_squat_rep_threshold(self, threshold_y):        
        self.squat_rep_threshold = threshold_y
        
    def track_marker(self, barbell_position):
        if (len(self.marker_history) >= self.history_length):
            self.marker_history.pop(0)
        self.marker_history.append(barbell_position)
        