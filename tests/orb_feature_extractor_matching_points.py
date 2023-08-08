import numpy as np
import cv2 as cv
from utils.helper_functions import json_reader, scale_intrinsic_matrix, create_intrinsic_matrix, match_points
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import utils.screen_capture as sc

# path =r'C:\Users\omri_\OneDrive\Documents\repos\gyroflow\resources\camera_presets\GoPro\GoPro_HERO11 Black Mini_Wide_16by9.json'
path =r'C:\Users\omri_\OneDrive\Documents\repos\gyroflow\resources\camera_presets\GoPro\GoPro_HERO9 Black_Wide_16by9.json'
camera_settings = json_reader(path)
intrinsic_matrix = np.array(camera_settings['fisheye_params']['camera_matrix'])
calib_resolution = tuple(camera_settings['calib_dimension'].values()) # (width, height)
n_grid_cells = 8
hfov = 120
max_disparity = 47 # 47 gives the best results
if n_grid_cells <= 4:
    min_disparity = 28
    N = 5000
else:
    min_disparity = 8
    N = 20000
n_neighbors = 8
min_n_matches = 20
max_matches_per_cell = 30 # -1 means no limit
max_neighbor_distance = 8
marker_size = 10

is_get_depth = False
is_draw_lines = False
is_draw_keypoints = True
cap = sc.ScreenCapture(monitor_number=1, tlwh=sc.YOUTUBE_TLWH_SMALL)
img = cap.capture()
v_fov = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(hfov) / 2) * img.shape[0] / img.shape[1]))
# K = scale_intrinsic_matrix(intrinsic_matrix, calib_resolution, img.shape[:2])
K = create_intrinsic_matrix(*img.shape[:2][::-1], hfov, vfov=v_fov)
focal = K[0, 0]
pp = (K[0, 2], K[1, 2])
p = 0.25
# define the size of the grid
grid_size = (n_grid_cells, n_grid_cells)
# create empty list with size of the grid
grid_prev_kp = [[[None] for _ in range(grid_size[0])] for _ in range(grid_size[1])]
grid_prev_des = [[[None] for _ in range(grid_size[0])] for _ in range(grid_size[1])]

cv.namedWindow('ORB Detection Test', cv.WINDOW_NORMAL)

def f(x=None):
    return


cv.createTrackbar('Max Features', 'ORB Detection Test', 500, N, f)
cv.createTrackbar('Scale Factor (x10)', 'ORB Detection Test', 20, 40, f)
cv.createTrackbar('Levels', 'ORB Detection Test', 8, 20, f)
cv.createTrackbar('WTA_K (2 or 4)', 'ORB Detection Test', 2, 4, f)

# Create BFMatcher object
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# FLANN parameters for ORB
# FLANN_INDEX_LSH = 6
# index_params= dict(algorithm = FLANN_INDEX_LSH,
#                    table_number = 6, # 12
#                    key_size = 12,     # 20
#                    multi_probe_level = 1) #2
# search_params = dict(checks=50)   # or pass empty dictionary
#
# matcher = cv.FlannBasedMatcher(index_params, search_params)

# Initialize variables for keypoints and descriptors
prev_kp, prev_des = None, None
velocity_dir = None
matches = None
pixel_coords, prev_pixel_coords = None, None

# init matplotlib figure
fig, ax = plt.subplots(1, 1)

while True:
    img = cap.capture()
    height, width = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # divide the image into grid
    cell_width = width // grid_size[0]
    cell_height = height // grid_size[1]

    nfeatures = cv.getTrackbarPos('Max Features', 'ORB Detection Test')
    nfeatures = 10 if nfeatures == 0 else nfeatures
    scaleFactor = cv.getTrackbarPos('Scale Factor (x10)', 'ORB Detection Test') / 10.0
    nlevels = cv.getTrackbarPos('Levels', 'ORB Detection Test')
    WTA_K = cv.getTrackbarPos('WTA_K (2 or 4)', 'ORB Detection Test')
    WTA_K = 2 if WTA_K == 2 else 4

    pts1 = []
    pts2 = []

    # create ORB object and compute keypoints and descriptors
    orb = cv.ORB_create(nfeatures=nfeatures//np.prod(grid_size), scaleFactor=scaleFactor, nlevels=nlevels, WTA_K=WTA_K)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Compute keypoints and descriptors for each cell
            cell = gray[j * cell_height:(j + 1) * cell_height, i * cell_width:(i + 1) * cell_width]
            cell_kp, cell_des = orb.detectAndCompute(cell, None)

            # Adjust the keypoint positions
            for k in cell_kp:
                k.pt = (k.pt[0] + i * cell_width, k.pt[1] + j * cell_height)

            if len(grid_prev_kp[j][i]) > 1 and len(grid_prev_des[j][i]) > 1 and (len(cell_kp) >= min_n_matches//np.prod(grid_size) and len(cell_kp) > 0):
                matches = match_points(matcher, grid_prev_des[j][i], cell_des, min_disparity=min_disparity, max_disparity=max_disparity, n_neighbors=0)
                #sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)[:max_matches_per_cell]
                pts1.extend(np.float32([grid_prev_kp[j][i][m.queryIdx].pt for m in matches]).reshape(-1, 2))
                pts2.extend(np.float32([cell_kp[m.trainIdx].pt for m in matches]).reshape(-1, 2))
                # assign cell_kp to grid_prev_kp

            grid_prev_kp[j][i] = cell_kp
            grid_prev_des[j][i] = cell_des

            # if len(cell_kp) > 0:
            #     kp.extend(cell_kp)
            #     des.extend(cell_des)

    if len(pts1) > 0:
        pts1 = np.vstack(pts1)
        pts2 = np.vstack(pts2)
    # if prev_kp is not None and prev_des is not None and len(des) > min_n_matches:
    if len(pts1) > min_n_matches:
        # matches = match_points(matcher, prev_des, des, min_disparity=8, max_disparity=47, n_neighbors=0)

        # create a new kp list and des list for the matched keypoints
        # matches = flann.match(prev_des, des)
        # matches = sorted(matches, key=lambda x: x.distance)[:100]
        # exclude matches with large disparity and also matches with extremely close disparity
        # matches = [m for m in matches if min_disparity < abs(prev_kp[m.queryIdx].pt[0] - kp[m.trainIdx].pt[0]) < max_disparity]
        # Extract the matched keypoints
        # pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        # pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Compute the essential matrix using the RANSAC algorithm
        # if len(matches) > min_n_matches:
        #     # clear figure
        #     ax.clear()
        #     # create a histogram of the distances between the matched keypoints
        #     distances = [m.distance for m in matches]
        #     ax.hist(distances, bins=max_disparity-min_disparity)
        #     ax.set_xlim(0, max_disparity+min_disparity)
        #     ax.set_ylim(0, 25)
        #     ax.set_title('Histogram of Keypoint Distances')
        #     ax.set_xlabel('Distance')
        #     ax.set_ylabel('Frequency')
        #     fig.canvas.draw()
        #     plt.pause(0.0000000001)


        # R = np.eye(3)
        # t = np.zeros((3, 1))

        # find essential matrix with 8-point algorithm
        # E, mask = cv.findEssentialMat(pts1, pts2, focal, pp, cv.FM_8POINT)
        E, mask = cv.findEssentialMat(pts1, pts2, focal, pp, cv.RANSAC, 0.99999999, 1.0, None)
        # Optionally, filter matches using the computed fundamental matrix
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        if len(pts1) > 0:
            _, R, t, _ = cv.recoverPose(E, pts1, pts2)
            # _, R, t, _ = cv.recoverPose(E, pts1, pts2)
            if is_get_depth:
                extrinsic = np.hstack([R, t])
                P1 = np.dot(K, np.eye(3, 4))
                P2 = np.dot(K, extrinsic)
                homogeneous_3D = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
                dehomo_3D = (homogeneous_3D / homogeneous_3D[3]).T
                depths = dehomo_3D[:, 2]

        print(t.flatten())
        # t = np.array([[1, 0, 1]])
        # Compute the fundamental matrix using the normalized 8-point algorithm
        # _, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
        # F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT)

        # Project the direction vector to the image plane
        # t = t / np.linalg.norm(t)
        velocity_dir = R.dot(t.reshape(3, 1))
        velocity_dir = velocity_dir / np.linalg.norm(velocity_dir)
        pixel_coords_hom = np.dot(K, velocity_dir)
        # pixel_coords_hom = np.dot(K, t.reshape(3, 1))
        pixel_coords = (pixel_coords_hom[0:2] / pixel_coords_hom[2])
        # # Convert to integer coordinates
        # if pixel_coords is within the image bounds:
        pixel_coords = (pixel_coords).astype(int).flatten()
        if len(pixel_coords) == 4:
            pixel_coords = [pixel_coords[0], pixel_coords[2]]

        # clamp pixel_coords to the image size
        pixel_coords[0] = max(0, min(pixel_coords[0], width))
        pixel_coords[1] = max(0, min(pixel_coords[1], height))
        print(pixel_coords)

        # Draw the cross at the projected point
        cross_size = 20
        color_white = (255, 255, 255)
        color_gray = (200, 200, 200)
        color_black = (0, 0, 0)
        thickness = 2
        radius = 2
        outer_radius = 5
        # draw a cross in the middle of the screen
        # img = cv.line(img, (width // 2 - cross_size, height // 2),
        #               (width // 2 + cross_size, height // 2), color, thickness)
        # img = cv.line(img, (width // 2, height // 2 - cross_size),
        #                 (width // 2, height // 2 + cross_size), color, thickness)

        # draw an X that passes through the center of the screen
        img = cv.line(img, (0, 0), (width, height), color_gray, 1)
        img = cv.line(img, (width, 0), (0, height), color_gray, 1)


        # draw a cross at the middle of the screen but without its center. make it out of 4 lines
        img = cv.line(img, (width // 2 - cross_size, height // 2), (width // 2 - radius - outer_radius, height // 2), color_black, thickness+2)
        img = cv.line(img, (width // 2 + cross_size, height // 2), (width // 2 + radius + outer_radius, height // 2), color_black, thickness+2)
        img = cv.line(img, (width // 2, height // 2 - cross_size), (width // 2, height // 2 - radius - outer_radius), color_black, thickness+2)
        img = cv.line(img, (width // 2, height // 2 + cross_size), (width // 2, height // 2 + radius + outer_radius), color_black, thickness+2)

        img = cv.line(img, (width // 2 - cross_size, height // 2), (width // 2 - radius - outer_radius, height // 2), color_white, thickness)
        img = cv.line(img, (width // 2 + cross_size, height // 2), (width // 2 + radius + outer_radius, height // 2), color_white, thickness)
        img = cv.line(img, (width // 2, height // 2 - cross_size), (width // 2, height // 2 - radius - outer_radius), color_white, thickness)
        img = cv.line(img, (width // 2, height // 2 + cross_size), (width // 2, height // 2 + radius + outer_radius), color_white, thickness)


        # draw a point in the middle of the screen
        img = cv.circle(img, (width // 2, height // 2), radius, color_black, thickness+1)
        img = cv.circle(img, (width // 2, height // 2), radius-1, color_white, thickness)

        # draw a circle at the projected point
        if 0 <= pixel_coords[0] < width and 0 <= pixel_coords[1] < height:
            if prev_pixel_coords is None: # first frame
                prev_pixel_coords = np.array(pixel_coords)
            else:
                pixel_coords = (p * np.array(pixel_coords) + (1 - p) * prev_pixel_coords).astype(int)
                prev_pixel_coords = pixel_coords
            # marker_size = int(1000 / len(pts2))
            # marker_size = max(3, marker_size)
            img = cv.circle(img, (pixel_coords[0], pixel_coords[1]), marker_size, color_black, thickness+2)
            img = cv.circle(img, (pixel_coords[0], pixel_coords[1]), marker_size, color_white, thickness)

        # img = cv.drawMatches(img, prev_kp, img, kp, matches[:500], None, flags=2)

    # img *= 0

    if is_draw_keypoints:
        # img = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))
        # draw keypoints that are matched with the previous frame
        if matches is not None:
            if len(pts2) > min_n_matches:
                # matched_kp = [kp[m.trainIdx] for m in matches]
                # img = cv.drawKeypoints(img, matched_kp, None, color=(0, 255, 0))
                # img = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))
                # draw a line between each keypoint and its maching keypoint from the previous frame
                # Loop over the matches and draw lines between matching points
                for pt1, pt2 in zip(pts1, pts2):
                    pt1 = tuple(np.round(pt1).astype(int))
                    pt2 = tuple(np.round(pt2).astype(int))

                    # Draw line in red color with thickness 1 px
                    cv.line(img, pt1, pt2, (0, 0, 255), 1)
                    # write text of the distance between the points next to the line
                    # cv.putText(img, str(np.linalg.norm(np.array(pt1) - np.array(pt2)).astype(np.int8)), pt2, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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
    if len(pts2)>0:
        # add text to image with len(matches)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, str(len(pts2)), (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow('ORB Detection Test', img)
    # if matches is not None and len(matches) > 100*min_n_matches:
    #     kp = [kp[m.trainIdx] for m in matches]
    #     des = np.vstack([des[m.trainIdx] for m in matches])
    # prev_kp, prev_des = kp, des

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
cap.close()