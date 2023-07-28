import numpy as np
import cv2 as cv
from sklearn.neighbors import NearestNeighbors

import utils.screen_capture as sc

n_neighbors = 2
max_neighbor_distance = 8
is_draw_lines = False
is_draw_keypoints = True
cap = sc.ScreenCapture(monitor_number=1, tlwh=sc.YOUTUBE_TLWH_SMALL)
img = cap.capture()
cv.namedWindow('ORB Detection Test', cv.WINDOW_NORMAL)

N = 5000
# create rainbow color palette with N colors:
# palette_prev = np.uint8([[[i * 255 / N, 255, 255] for i in range(N)]])[0]
#random palette
palette_prev = np.random.randint(0, 255, (N, 3)).astype(np.uint8)
palette = np.uint8([[[i * 255 / N, 255, 255] for i in range(N)]])[0]

def f(x=None):
    return


cv.createTrackbar('Max Features', 'ORB Detection Test', 500, N, f)
cv.createTrackbar('Scale Factor (x10)', 'ORB Detection Test', 20, 40, f)
cv.createTrackbar('Levels', 'ORB Detection Test', 8, 20, f)
cv.createTrackbar('WTA_K (2 or 4)', 'ORB Detection Test', 2, 4, f)

# Create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# FLANN parameters for ORB
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)

# Initialize variables for keypoints and descriptors
prev_kp, prev_des = None, None

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

    if prev_kp is not None and prev_des is not None:
        # matches = bf.match(prev_des, des)
        matches = flann.match(prev_des, des)
        print(len(matches))
        # for match in matches:
        #     prev_des_index = match.queryIdx
        #     des_index = match.trainIdx
        #     img = cv.drawKeypoints(img, [kp[des_index]], None, color=tuple(palette_prev[prev_des_index].tolist()))
        #     # change palette color according to pallette_prev:
        #     palette[des_index] = palette_prev[prev_des_index]
        # palette_prev = palette.copy()


        # img = cv.drawMatches(img, prev_kp, img, kp, matches[:500], None, flags=2)

    # img *= 0

    if is_draw_keypoints:
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
                    img = cv.line(img, tuple(np.int0(kp_array[i])), tuple(np.int0(kp_array[neighbor_index])),
                                  (0, 255, 0), 1)

    cv.imshow('ORB Detection Test', img)

    prev_kp, prev_des = kp, des

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
cap.close()