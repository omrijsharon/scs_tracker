import numpy as np
import cv2 as cv
from sklearn.neighbors import NearestNeighbors

import utils.screen_capture as sc

n_neighbors = 2
max_neighbor_distance = 8
is_draw_lines = True
is_draw_keypoints = not is_draw_lines
cap = sc.ScreenCapture(monitor_number=1, tlwh=sc.YOUTUBE_TLWH_SMALL)
img = cap.capture()
cv.namedWindow('ORB Detection Test', cv.WINDOW_NORMAL)

def f(x=None):
    return

cv.createTrackbar('Max Features', 'ORB Detection Test', 500, 10000, f)
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

    if is_draw_keypoints:
        # img = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 0))
        img = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))
    if is_draw_lines:
        # Convert keypoints to an array of their coordinates
        kp_array = np.array([point.pt for point in kp])

        # Use sklearn's NearestNeighbors to find the nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(kp_array)
        distances, indices = neighbors.kneighbors(kp_array)

        # Draw lines between each point and its two nearest neighbors
        for i, point in enumerate(kp):
            for neighbor_index, neighbor_distance in zip(indices[i][1:], distances[i][1:]):  # Skip the point itself, which is always the first neighbor
                if neighbor_distance <= max_neighbor_distance:  # Add your distance check here
                    img = cv.line(img, tuple(np.int0(kp_array[i])), tuple(np.int0(kp_array[neighbor_index])), (0, 255, 0), 1)

    cv.imshow('ORB Detection Test', img)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
cap.close()