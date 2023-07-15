import numpy as np
import cv2 as cv

import utils.screen_capture as sc

cap = sc.ScreenCapture(monitor_number=1, tlwh=sc.YOUTUBE_TLWH_SMALL)
img = cap.capture()

cv.namedWindow('FAST Corner Detection Test', cv.WINDOW_NORMAL)

def f(x=None):
    return

cv.createTrackbar('Threshold', 'FAST Corner Detection Test', 10, 100, f)

while True:
    img = cap.capture()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    threshold = cv.getTrackbarPos('Threshold', 'FAST Corner Detection Test')

    if threshold <= 0:
        threshold = 1

    # Initiate FAST detector
    fast = cv.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)

    img *= 0
    # Find and draw the keypoints
    keypoints = fast.detect(gray, None)
    img2 = cv.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    cv.imshow('FAST Corner Detection Test', img2)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
cap.close()