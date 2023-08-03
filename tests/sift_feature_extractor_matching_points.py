import numpy as np
import cv2 as cv
from utils.helper_functions import json_reader, scale_intrinsic_matrix, create_intrinsic_matrix
from sklearn.neighbors import NearestNeighbors

import utils.screen_capture as sc

# path =r'C:\Users\omri_\OneDrive\Documents\repos\gyroflow\resources\camera_presets\GoPro\GoPro_HERO11 Black Mini_Wide_16by9.json'
path =r'C:\Users\omri_\OneDrive\Documents\repos\gyroflow\resources\camera_presets\GoPro\GoPro_HERO9 Black_Wide_16by9.json'
camera_settings = json_reader(path)
intrinsic_matrix = np.array(camera_settings['fisheye_params']['camera_matrix'])
calib_resolution = tuple(camera_settings['calib_dimension'].values()) # (width, height)
hfov = 150
min_disparity = 45
max_disparity = 1
n_neighbors = 2
max_neighbor_distance = 8
is_draw_lines = False
is_draw_keypoints = True
cap = sc.ScreenCapture(monitor_number=1, tlwh=sc.YOUTUBE_TLWH_SMALL)
img = cap.capture()
v_fov = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(hfov) / 2) * img.shape[0] / img.shape[1]))
# K = scale_intrinsic_matrix(intrinsic_matrix, calib_resolution, img.shape[:2])
K = create_intrinsic_matrix(*img.shape[:2][::-1], hfov, vfov=v_fov)
focal = K[0, 0]
pp = (K[0, 2], K[1, 2])

cv.namedWindow('SIFT Detection Test', cv.WINDOW_NORMAL)

detector = cv.SIFT_create()

N = 10000
# create rainbow color palette with N colors:
# palette_prev = np.uint8([[[i * 255 / N, 255, 255] for i in range(N)]])[0]
#random palette
# palette_prev = np.random.randint(0, 255, (N, 3)).astype(np.uint8)
# palette = np.uint8([[[i * 255 / N, 255, 255] for i in range(N)]])[0]

def f(x=None):
    return


cv.createTrackbar('Max Features', 'SIFT Detection Test', 500, N, f)


# Create BFMatcher object
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)  # This is for SIFT

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Initialize variables for keypoints and descriptors
prev_kp, prev_des = None, None
matches = None
pixel_coords_prev = None
p = 0.05

while True:
    img = cap.capture()
    height, width = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    nfeatures = cv.getTrackbarPos('Max Features', 'SIFT Detection Test')

    kp, des = detector.detectAndCompute(gray, None)
    kp = sorted(kp, key=lambda x: -x.response)[:nfeatures]  # Keep only top N keypoints
    des = detector.compute(gray, kp)[1]  # Recompute descriptors for top N keypoints

    if prev_kp is not None and prev_des is not None:
        matches = flann.match(prev_des, des)

        # Sort matches by distance (smaller is better)
        matches = sorted(matches, key=lambda x: x.distance)
        # matches = flann.match(prev_des, des)
        # matches = [m for m in matches if abs(prev_kp[m.queryIdx].pt[0] - kp[m.trainIdx].pt[0]) > min_disparity]
        # matches = sorted(matches, key=lambda x: x.distance)[:100]
        # exclude matches with large disparity and also matches with extremely close disparity
        # matches = [m for m in matches if max_disparity < abs(prev_kp[m.queryIdx].pt[0] - kp[m.trainIdx].pt[0]) < min_disparity]
        matches = [m for m in matches if max_disparity < m.distance < min_disparity]
        # Extract the matched keypoints
        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2).astype(np.float64)
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2).astype(np.float64)

        # Compute the essential matrix using the RANSAC algorithm

        if len(matches) > 10:
            # R = np.eye(3)
            # t = np.zeros((3, 1))

            # find essential matrix with 8-point algorithm
            # E, mask = cv.findEssentialMat(pts1, pts2, focal, pp, cv.FM_8POINT)
            E, mask = cv.findEssentialMat(pts2, pts1, focal, pp, cv.RANSAC, 0.99, 3.0, None)
            # Optionally, filter matches using the computed fundamental matrix
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

            if len(pts1) > 0:
                _, R, t, _ = cv.recoverPose(E, pts1, pts2)
            print(t.flatten())
            # t = np.array([[1, 0, 1]])
            # Compute the fundamental matrix using the normalized 8-point algorithm
            # _, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
            # F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT)




            # F, _ = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT)
            # print("F:", F)
            # # # SVD of the fundamental matrix
            # U, D, Vt = np.linalg.svd(F)
            # print("Vt:", Vt)
            # # The right singular vector corresponding to the smallest singular value gives the epipole in the second image
            # # e2 = np.array([1, 0, 1])
            # e2 = Vt[-1, :]
            # e2 = e2 / e2[2]
            # e2 = e2/np.linalg.norm(e2)
            #
            # # Calculate field of view in radians
            # hfov_rad = np.radians(hfov)
            # vfov_rad = hfov_rad * height / width
            #
            # # Compute the normalized direction vector in camera coordinates
            # dir_vector = e2
            # dir_vector = np.array([e2[0], e2[1], 1])
            #
            # # Normalize the direction vector
            # dir_vector = dir_vector / np.linalg.norm(dir_vector)



            # Project the direction vector to the image plane
            t = t / np.linalg.norm(t)
            pixel_coords_hom = np.dot(K, R.dot(t.reshape(3, 1)))
            # pixel_coords_hom = np.dot(K, t.reshape(3, 1))
            pixel_coords_curret = (pixel_coords_hom[0:2] / pixel_coords_hom[2])
            # # Convert to integer coordinates
            pixel_coords_curret = (pixel_coords_curret).astype(int).flatten()
            print(pixel_coords_curret)
            if pixel_coords_prev is None:
                pixel_coords_prev = pixel_coords_curret
            pixel_coords = p * pixel_coords_curret + (1-p) * pixel_coords_prev

            # Draw the cross at the projected point
            cross_size = 20
            color = (0, 255, 0)  # green color
            thickness = 2
            # draw a cross in the middle of the screen
            img = cv.line(img, (width // 2 - cross_size, height // 2),
                          (width // 2 + cross_size, height // 2), color, thickness)
            img = cv.line(img, (width // 2, height // 2 - cross_size),
                            (width // 2, height // 2 + cross_size), color, thickness)
            # draw a circle at the projected point
            img = cv.circle(img, (pixel_coords[0], pixel_coords[1]), 10, color, thickness)
            pixel_coords_prev = pixel_coords
        # print(len(matches))
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
        # draw keypoints that are matched with the previous frame
        if matches is not None:
            matched_kp = [kp[m.trainIdx] for m in matches]
            img = cv.drawKeypoints(img, matched_kp, None, color=(0, 255, 0))
            # img = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))
            # draw a line between each keypoint and its maching keypoint from the previous frame
            if matches is not None:
                # Loop over the matches and draw lines between matching points
                for pt1, pt2 in zip(pts1, pts2):
                    pt1 = tuple(np.round(pt1).astype(int))
                    pt2 = tuple(np.round(pt2).astype(int))

                    # Draw line in red color with thickness 1 px
                    cv.line(img, pt1, pt2, (0, 0, 255), 1)
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

    cv.imshow('SIFT Detection Test', img)

    prev_kp, prev_des = kp, des

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
cap.close()