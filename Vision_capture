import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0);
#(grabbed, frame) = camera.read()


cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:
    #frame = cv2.imread('smarties.png')
    _, frame = cap.read()

    #Gaussian filter
    GaussBlur = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(GaussBlur, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    #Kernel

    kernel = np.ones((1, 1), np.uint8)

    #Mask

    mask = cv2.inRange(hsv, l_b, u_b)
    #ret, thresh = cv2.threshold(GaussBlur, l_b, u_b, 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_NONE)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
