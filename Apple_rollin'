import numpy as np
import cv2 as cv2
import imutils

lowerApple = (166, 84, 141)
upperApple = (186, 255, 255)

# Initialize the points
u_x = 0
u_y = 0

video = cv2.VideoCapture('app_move.mp4')

frame_width = int(video.get(3))
frame_height = int(video.get(4))

out = cv2.VideoWriter('apple_Kalman.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

##Kalman variables

def predict(x_k, P, u, A):
    x_k = np.matmul(A, x_k) + u
    P = np.matmul(A, np.matmul(P, A.transpose())) + Q
    return [x_k, P]

def update(x_k, P, Z, H, R):
    y = Z - np.matmul(H, x_k)
    S = np.matmul(H, np.matmul(P, H.transpose())) + R
    K = np.matmul(P, np.matmul(H.transpose(), np.linalg.pinv(S)))
    x_k = x_k + np.matmul(K, y)
    P = np.matmul((I - np.matmul(K, H)), P)
    #print("K is", K[0][0])

    return [x_k, P]

x1 = np.array(np.mat('0.0; 0.0; 0.0; 0.0'))  # initial location and velocity
P = np.array(np.mat('100, 0, 0, 0; 0, 100, 0, 0; 0, 0, 100, 0; 0, 0, 0, 10'))  # initial uncertainty
u = np.array(np.mat('0;0;0;0'))  # control input
A = np.array(np.mat('1, 0, 0.1, 0; 0, 1, 0, 0.1; 0, 0, 1, 0; 0, 0, 0, 1'))  # state matrix
H = np.array(np.mat('1, 0, 0, 0; 0, 1, 0, 0'))  # measurement
R = np.array(np.mat('100, 0; 0, 100'))  # measurement uncertainty, good values
Q = np.array(np.mat('1000,0,0,0;0,100,0,0;0,0,11000,0;0,0,0,100'))  # estimation uncertainty
I = np.array(np.mat('1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1')) #identity matrix

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

while True:
    check, frame = video.read()

    if check is None:
        break

    #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    blurred = cv2.medianBlur(frame, 3)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.inRange(hsv, lowerApple, upperApple)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 2:
                cv2.circle(frame, (int(x), int(y)), int(radius), 35, 2, 8)

                p = Point(int(x), int(y))
                u_x = p.x
                u_y = p.y

                x1, P = update(x1, P, np.mat('{}; {}'.format(u_x, u_y)), H, R)
                x1, P = predict(x1, P, u, A)
                KalmanPred = x1

                # trial
                predx = KalmanPred[0]
                predy = KalmanPred[1]
                cv2.circle(frame, (KalmanPred[0], KalmanPred[1]), 90, [0, 255, 255], 2, 8)

        else:
            x1, P = predict(x1, P, u, A)
            KalmanPred = x1
            cv2.circle(frame, (KalmanPred[0], KalmanPred[1]), 90, [0, 255, 255], 2, 8)

    out.write(frame)
    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == 27:
        break


video.release()
out.release()
cv2.destroyAllWindows()
