import numpy as np
import cv2 as cv
import utils.screen_capture as sc

cap = sc.ScreenCapture(monitor_number=1, tlwh=sc.YOUTUBE_TLWH_SMALL)
img = cap.capture()
cv.namedWindow('ORB Detection Test', cv.WINDOW_NORMAL)

def f(x=None):
    return

cv.createTrackbar('Max Features', 'ORB Detection Test', 500, 5000, f)
cv.createTrackbar('Scale Factor (x10)', 'ORB Detection Test', 20, 40, f)
cv.createTrackbar('Levels', 'ORB Detection Test', 8, 20, f)
cv.createTrackbar('WTA_K (2 or 4)', 'ORB Detection Test', 2, 4, f)

while True:
    img = cap.capture()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    nfeatures = cv.getTrackbarPos('Max Features', 'ORB Detection Test')
    scaleFactor = cv.getTrackbarPos('Scale Factor (x10)', 'ORB Detection Test') / 10.0
    nlevels = cv.getTrackbarPos('Levels', 'ORB Detection Test')
    WTA_K = cv.getTrackbarPos('WTA_K (2 or 4)', 'ORB Detection Test')
    WTA_K = 2 if WTA_K == 2 else 4

    # create ORB object and compute keypoints and descriptors
    orb = cv.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels, WTA_K=WTA_K)
    kp, des = orb.detectAndCompute(gray, None)

    img *= 0

    # draw keypoints
    # img = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 0))
    img = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))


    cv.imshow('ORB Detection Test', img)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
cap.close()