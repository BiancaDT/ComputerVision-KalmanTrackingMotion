import numpy as np
import cv2 as cv2
import imutils

video = cv2.VideoCapture("VIDA.mp4")

ret, first_frame = video.read()

yellowLower = (0, 90, 80)
yellowUpper = (30, 255, 255)

# Initialize the points
u_x = 0
u_y = 0

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
    print("K is", K[0][0])

    return [x_k, P]


x1 = np.array(np.mat('0.0; 580.0; 0.0; 0.0'))  # initial location and velocity
P = np.array(np.mat('100, 0, 0, 0; 0, 100, 0, 0; 0, 0, 100, 0; 0, 0, 0, 10'))  # initial uncertainty
u = np.array(np.mat('0;0;10;0'))  # control input
A = np.array(np.mat('1, 0, 0.5, 0; 0, 1, 0, 0.1; 0, 0, 1, 0; 0, 0, 0, 1'))  # state matrix
H = np.array(np.mat('1, 0, 0, 0; 0, 1, 0, 0'))  # measurement
R = np.array(np.mat('10, 0; 0, 100'))  # measurement uncertainty, good values
Q = np.array(np.mat('1000,0,0,0;0,100,0,0;0,0,500,0;0,0,0,100'))  # estimation uncertainty
I = np.array(np.mat('1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1')) #identity matrix

#Q estimate of position needs to be < 100

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

while True:
    check, frame = video.read()

    if check is None:
        break

    GaussFilter = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(GaussFilter, cv2.COLOR_BGR2HSV)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        # rect = (center(x, y), (width, height), angle of rotation)
        rect = cv2.minAreaRect(c)
        # ((x, y)) = cv2.minAreaRect(c)
        # box is the 4 corners of rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        x = rect[0][0]
        y = rect[0][1]
        width = rect[1][0]
        height = rect[1][1]

        if width > 0:
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            # cv2.putText(frame, "thing", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)
            p = Point(int(x), int(y))
            u_x = p.x
            u_y = p.y

            x1, P = update(x1, P, np.mat('{}; {}'.format(u_x, u_y)), H, R)
            x1, P = predict(x1, P, u, A)
            KalmanPred = x1

            #trial
            predx = KalmanPred[0]
            predy = KalmanPred[1]
            cv2.circle(frame, (KalmanPred[0], KalmanPred[1]), 35, [100, 100, 30], 2, 8)

    else:
        x1, P = predict(x1, P, u, A)
        KalmanPred = x1
        #print("Pred: ", KalmanPred)
        #print("Pred x: ", KalmanPred[0])
        #print("Pred y: ", KalmanPred[1])
        predx = KalmanPred[0]
        predy = KalmanPred[1]
        pred = (predx, predy)

        # Center, radius
        # for rectangle, x, then x plus width minus y
        cv2.circle(frame, (KalmanPred[0], KalmanPred[1]), 35, [100, 100, 30], 2, 8)
        V2_rect = predx+width-height
        #print(V2_rect)

    cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)

    if cv2.waitKey(300) == 27:
        break

video.release()
cv2.destroyAllWindows()
