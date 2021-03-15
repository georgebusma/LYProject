import cv2 as cv
import numpy as np
from tracker import *

#Create tracker object
tracker = EuclideanDistTracker()

cap = cv.VideoCapture("bettervideo_Trim.mp4")

#Object detection
obj_KNN = cv.createBackgroundSubtractorKNN()
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

#Initialise the fruit counter
fruit = 0

def empty(a):
    pass

#Creates a window with Trackbars to adjust the Hue, Saturation and Value of the video
cv.namedWindow("HSV")
cv.resizeWindow("HSV", 640, 640)
cv.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv.createTrackbar("HUE Max", "HSV", 15, 179, empty)
cv.createTrackbar("SAT Min", "HSV", 150, 255, empty)
cv.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv.createTrackbar("VALUE Min", "HSV", 35, 255, empty)
cv.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

while True:
    ret, frame = cap.read()

    #Resize the frames
    scale_percent = 44 #percent of original size
    width = int(frame.shape[1] * scale_percent/100)
    height = int(frame.shape[0] * scale_percent/100)
    dim = (width, height)
    frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    # print(height, width)

    #Extracting Region of interest
    roi = frame[0:580, 0:260]


    #Using HSV Colorspace for fruit detection
    imgHSV = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos("HUE Min", "HSV")
    h_max = cv.getTrackbarPos("HUE Max", "HSV")
    s_min = cv.getTrackbarPos("SAT Min", "HSV")
    s_max = cv.getTrackbarPos("SAT Max", "HSV")
    v_min = cv.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv.getTrackbarPos("VALUE Max", "HSV")
    # print(h_min)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(imgHSV, lower, upper)
    result = cv.bitwise_and(roi, roi, mask=mask)
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    foreMask = object_detector.apply(result) #extract the foreground mask

    # Draw the reference traffic lines
    cv.line(frame, (238, 0), (238, 580), (255, 0, 255), 1)  # Violet line
    cv.line(roi, (235, 0), (235, 580), (255, 255, 0), 1)
    # cv.line(roi, (258, 0), (268, 580), (255, 255, 0), 2)

    #1. Object detection

    mask = cv.cvtColor(result, cv.COLOR_HSV2BGR)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    # _, mask = cv.threshold(mask, 250, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    validFruits = []
    detected = []
    for cnt in contours:
        #Calculate area and remove small elements
        area = cv.contourArea(cnt)
        cv.fillPoly(mask, pts=[cnt], color=(255, 255, 255))
        if area > 200:
            # cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)

            x, y, w, h = cv.boundingRect(cnt)
            validated_contour = (w >= 25 and w <= 50) and (h >= 25 and h <= 50)
            if not validated_contour:
                continue

            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            centre = (x + int(w/2), y + int(h/2))

            validFruits.append(centre)
            detected.append([x, y, w, h])
            cv.circle(frame, (x + int(w/2), y + int(h/2)), 2, (250, 200, 200), -1)

            # Count fruits that crossed the line
            for (x, y) in validFruits:
                if (x, y) not in validFruits:
                    continue
                else:
                    if x > 235:
                        if x < 238:
                            fruit += 1
                            cv.line(frame, (238, 0), (238, 580), (50, 0, 255), 2)  # Violet line turn red when fruits cross it
                            validFruits.remove((x, y))
                            print("Fruit detected: " + str(fruit))




    # 2. Object Tracking
    # boxes_ids = tracker.update(detected)
    # for id in boxes_ids:
    #     x, y, w, h, id = id
    #     cv.putText(frame, str(id), (x-15, y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0,0), 2)
    #     cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # cv.imshow("roi", roi)
    cv.imshow("frame", frame)
    cv.imshow("mask", mask)
    cv.imshow("sub", foreMask)
    cv.imshow("result", result)

    key = cv.waitKey(0)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()