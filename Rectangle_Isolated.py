import numpy as np
import cv2 as cv2
import imutils
import matplotlib.pyplot as plt

video = cv2.VideoCapture("VIDA.mp4")
#video = cv2.VideoCapture("app_move.mp4")

ret, first_frame = video.read()

#first_frame = imutils.resize(first_frame, width=600)

#Positions
x = 6
y = 577
width = 108
height = 54

#Calculate the ROI
roi = first_frame[y: y+height, x: x+width]
#cv2.imshow("roi", roi)

#Change the ROI to HSV
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

#Calculate the color histogram of roi
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

#Normalize the histogram because we have some outliers
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#print(roi_hist)
frames_counter = 1

while True:
    frames_counter = frames_counter + 1
    check, frame = video.read()
    cv2.imshow("roi", roi)

    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


    if check:
        # Backprojection tells where the ROI is in the video
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Meanshift finds the area with concentration of white points
        _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
        print(track_window)
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x +w, y+h), (0, 255, 0), 2)

        cv2.imshow("Mask", mask)

        #lower_range = np.array([110, 50, 50])
        #upper_range = np.array([130, 255, 255])

        #blocks out in black everything not in the specified range
        #mask = cv2.inRange(hsv, lower_range, upper_range)

        #result = cv2.bitwise_and(frame, frame, mask = mask)

        #cv2.imshow("First Frame", first_frame)
        cv2.imshow("Frame", frame)
        #cv2.imshow("Masked", mask)
        #cv2.imshow("Result", result)

        key = cv2.waitKey(200)
    else:
        break

        #show a frame each 60 miliseconds
        #if cv2.waitKey(60) == 27:
            #break

print("Number of frames in the video: ", frames_counter)
video.release()
cv2.destroyAllWindows()


video.release()
cv2.destroyAllWindows()
