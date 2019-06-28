import numpy as np
import cv2 as cv2
import imutils
import matplotlib.pyplot as plt

video = cv2.VideoCapture("VIDA.mp4")
#video = cv2.VideoCapture("app_move.mp4")

ret, first_frame = video.read()

#first_frame = imutils.resize(first_frame, width=600)

class Point:
    def __init__(self, x=0, y=0, c=""):
        self.x = x
        self.y = y
        self.c = c

corx=0
cory=0


#Positions
x = 6
y = 577
width = 108
height = 54

#Calculate the ROI
roi = first_frame[y: y+height, x: x+width]
#cv2.imshow("roi", roi)

#GausBlur = cv2.GaussianBlur(roi, (11, 11), 0)

#Change the ROI to HSV
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

#Calculate the color histogram of roi
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

#Normalize the histogram because we have some outliers
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#print(roi_hist)
frames_counter = 1

#Kalman Filter

def predict(x_k, P, u, A): #F = A
    x_k = np.matmul(A, x_k) + u                            #State vector
    P = np.matmul(A, np.matmul(P, A.transpose())) + Rt #Uncertainty covariance - estimation error
    return [x_k, P]

def update(x_k, P, Z, H, R):
    y = Z - np.matmul(H, x_k)
    S = np.matmul(H,np.matmul(P, H.transpose())) + R
    K = np.matmul(P,np.matmul(H.transpose(), np.linalg.pinv(S))) #Kalman gain
    x_k = x_k + np.matmul(K, y)
    P = np.matmul((I - np.matmul(K, H)),P)

    return [x_k, P]

#Predict variables
#Vector format: [x, xdot, y, ydot], dt = 0.1

x_k = np.array(np.mat('6; 0.0; 577; 0.0'))           # initial state (location and velocity) - control matrix
P = np.array(np.mat('1000, 0, 0, 0; 0, 1000, 0, 0; 0, 0, 1000, 0; 0, 0, 0, 1000'))    # initial uncertainty
u =  np.array(np.mat('0;0;0;0'))                                # control input = acceleration - we assume no acceleration
A =  np.array(np.mat('1, 0.1, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0.1; 0, 0, 0, 1'))          # state transition matrix

#Update variables
H =  np.array(np.mat('1, 0, 0, 0; 0, 1, 0, 0'))                     # measurement function
Q =  np.array(np.mat('10000, 0; 0, 10000'))                                    # measurement uncertainty
R =  np.array(np.mat('10000,0,0,0;0,10000,0,0;0,0,10000,0;0,0,0,10000'))           # next state uncertainty
I =  np.array(np.mat('1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1'))

while True:
    frames_counter = frames_counter + 1
    check, frame = video.read()
    #cv2.imshow("roi", roi)

    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    if check:
        # Backprojection tells where the ROI is in the video
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        #Morph operations

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


        ret, track_window = cv2.CamShift(mask, (x, y, width, height), term_criteria)

        # Gives us the position and orientation
        # x, y, with, height, rotation
        #print(ret)

        # Points of position and orientation, made into a rectangle by this function
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)

        # Ret can only do rectilinear, no rotation, so we do rotations now
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

        p = Point(int(x), int(y), 255)

        #x_k, P = update(Z, x_k, P, H, R, I)

        #x_k, P = predict(x_k, P, u, A)
        #print(x_k)

        x, y, w, h = track_window

        cv2.imshow("Frame", frame)
        #cv2.imshow("Mask", mask)

        if cv2.waitKey(200) == 27:
            break
    else:
        break


print("Number of frames in the video: ", frames_counter)
video.release()
cv2.destroyAllWindows()
