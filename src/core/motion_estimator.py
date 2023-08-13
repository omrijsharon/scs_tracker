import numpy as np
import cv2 as cv
from utils.helper_functions import json_reader, scale_intrinsic_matrix, create_intrinsic_matrix, match_points, draw_osd
from time import perf_counter

import utils.screen_capture as sc

n_grid_cells = 8
min_n_matches = 9
max_disparity = 27
marker_size = 10
p = 0.5
is_draw_keypoints = True
matcher_type = 'bf' # or 'flann'

cross_size = 20
color_white = (255, 255, 255)
color_gray = (200, 200, 200)
color_black = (0, 0, 0)
thickness = 2
radius = 2
outer_radius = 5

cap = sc.ScreenCapture(monitor_number=1, tlwh=sc.YOUTUBE_TLWH_SMALL)

hfov = 120
cell_size = (cap.monitor["width"] / n_grid_cells, cap.monitor["height"] / n_grid_cells)
img = cap.capture()
v_fov = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(hfov) / 2) * img.shape[0] / img.shape[1]))
K = create_intrinsic_matrix(*img.shape[:2][::-1], hfov, vfov=v_fov)
focal = K[0, 0]
pp = (K[0, 2], K[1, 2])

# define the size of the grid
grid_size = np.array((n_grid_cells, n_grid_cells))
total_cells = np.prod(grid_size-2) # total number of cells in the grid without the outermost cells
height, width = img.shape[:2]
cell_width = width // grid_size[0]
cell_height = height // grid_size[1]
# create empty list with size of the grid
grid_prev_kp = [[None for _ in range(grid_size[0])] for _ in range(grid_size[1])]
grid_prev_des = [[None for _ in range(grid_size[0])] for _ in range(grid_size[1])]
grid_matches_prev_idx = [[None for _ in range(grid_size[0])] for _ in range(grid_size[1])]

cv.namedWindow('ORB Detection Test', cv.WINDOW_NORMAL)

def f(x=None):
    return

cv.createTrackbar('Max Features', 'ORB Detection Test', 2800, 5000, f)
cv.createTrackbar('Scale Factor (x10)', 'ORB Detection Test', 20, 40, f)
cv.createTrackbar('Levels', 'ORB Detection Test', 8, 20, f)
cv.createTrackbar('WTA_K (2 or 4)', 'ORB Detection Test', 2, 4, f)
cv.createTrackbar('edgeThreshold', 'ORB Detection Test', 1, 50, f)
cv.createTrackbar('patchSize', 'ORB Detection Test', 31, 100, f)
cv.createTrackbar('fastThreshold', 'ORB Detection Test', 40, 100, f)
cv.createTrackbar('RANSAC subsample_size', 'ORB Detection Test', 50, 250, f)
cv.createTrackbar('maxIters', 'ORB Detection Test', 40, 500, f)
cv.createTrackbar('Min Disparity', 'ORB Detection Test', 7, max_disparity, f)
cv.createTrackbar('Max Matches per Cell', 'ORB Detection Test', 0, 100, f)
cv.createTrackbar('Min Matches', 'ORB Detection Test', min_n_matches, 500, f)
cv.createTrackbar('p', 'ORB Detection Test', int(p * 100), 100, f)
cv.createTrackbar('draw keypoints?', 'ORB Detection Test', 1, 1, f)


if matcher_type == 'bf':
    # Create BFMatcher object
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
elif matcher_type == 'flann':
    # FLANN parameters for ORB
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
    search_params = dict(checks=50)   # or pass empty dictionary

    matcher = cv.FlannBasedMatcher(index_params, search_params)
else:
    raise ValueError('matcher_type must be "bf" or "flann"')

velocity_dir = None
matches = None
pixel_coords, prev_pixel_coords = None, None

while True:
    img = cap.capture()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    nfeatures = cv.getTrackbarPos('Max Features', 'ORB Detection Test')
    nfeatures = 10 if nfeatures == 0 else nfeatures
    scaleFactor = cv.getTrackbarPos('Scale Factor (x10)', 'ORB Detection Test') / 10.0
    nlevels = cv.getTrackbarPos('Levels', 'ORB Detection Test')
    WTA_K = cv.getTrackbarPos('WTA_K (2 or 4)', 'ORB Detection Test')
    WTA_K = 2 if WTA_K == 2 else 4
    edgeThreshold = cv.getTrackbarPos('edgeThreshold', 'ORB Detection Test')
    edgeThreshold = 31 if edgeThreshold == 0 else edgeThreshold
    patchSize = cv.getTrackbarPos('patchSize', 'ORB Detection Test')
    patchSize = 2 if patchSize <= 2 else patchSize
    fastThreshold = cv.getTrackbarPos('fastThreshold', 'ORB Detection Test')
    fastThreshold = 1 if fastThreshold == 0 else fastThreshold
    subsample_size = cv.getTrackbarPos('RANSAC subsample_size', 'ORB Detection Test')
    subsample_size = 10 if subsample_size <= 10 else subsample_size
    maxIters = cv.getTrackbarPos('maxIters', 'ORB Detection Test')
    maxIters = 1 if maxIters == 0 else maxIters
    min_disparity = cv.getTrackbarPos('Min Disparity', 'ORB Detection Test')
    max_matches_per_cell = cv.getTrackbarPos('Max Matches per Cell', 'ORB Detection Test')
    max_matches_per_cell = -1 if max_matches_per_cell == 0 else max_matches_per_cell
    min_n_matches = cv.getTrackbarPos('Min Matches', 'ORB Detection Test')
    min_n_matches = 10 if min_n_matches < 10 else min_n_matches
    p = cv.getTrackbarPos('p', 'ORB Detection Test') / 100.0
    p = 0.01 if p == 0 else p
    is_draw_keypoints = bool(cv.getTrackbarPos('draw keypoints?', 'ORB Detection Test'))

    pts1 = []
    pts2 = []
    t0 = perf_counter()
    orb = cv.ORB_create(nfeatures=nfeatures // total_cells, scaleFactor=scaleFactor, nlevels=nlevels, WTA_K=WTA_K, edgeThreshold=edgeThreshold, fastThreshold=fastThreshold, scoreType=cv.ORB_HARRIS_SCORE)
    t1 = perf_counter() - t0
    for i in range(1, grid_size[0]-1):
        for j in range(1, grid_size[1]-1):
            # Compute keypoints and descriptors for each cell
            cell = gray[j * cell_height:(j + 1) * cell_height, i * cell_width:(i + 1) * cell_width]
            cell_kp, cell_des = orb.detectAndCompute(cell, None)
            is_any_kp = len(cell_kp) > 0
            min_n_matches_per_cell = min_n_matches//(np.prod(np.array(grid_size)-2))
            is_kp_more_than_min_n_matches = len(cell_kp) >= min_n_matches_per_cell
            # Adjust the keypoint positions
            for k in cell_kp:
                k.pt = (k.pt[0] + i * cell_width, k.pt[1] + j * cell_height)
                # k.pt = tuple(map(sum, zip(k.pt, (i * cell_width, j * cell_height))))
            if grid_prev_kp[j][i] is not None: # not first frame
                if len(grid_prev_kp[j][i]) > 1 and len(grid_prev_des[j][i]) > 1 and is_kp_more_than_min_n_matches and is_any_kp:
                    if grid_matches_prev_idx[j][i] is None: # first pair of frames
                        prev_des = grid_prev_des[j][i]
                    else:
                        # check if all the indices in grid_matches_prev_idx[j][i] are valid
                        valid_matches = ([0 <= idx < len(grid_prev_des[j][i]) for idx in grid_matches_prev_idx[j][i]])
                        if len(grid_matches_prev_idx[j][i]) <= min_n_matches_per_cell or any(valid_matches):
                            prev_des = grid_prev_des[j][i]
                        else:
                            grid_matches_prev_idx[j][i] = np.array(grid_matches_prev_idx[j][i])[valid_matches]
                            prev_des = np.take(grid_prev_des[j][i], grid_matches_prev_idx[j][i], axis=0)
                    # matches = match_points(matcher, prev_des, cell_des, min_disparity=min_disparity, max_disparity=max_disparity, n_neighbors=0)
                    matches = matcher.match(prev_des, cell_des)
                    matches = sorted(matches, key=lambda x: x.distance)[:max_matches_per_cell]
                    pts1.extend(np.float32([grid_prev_kp[j][i][m.queryIdx].pt for m in matches]).reshape(-1, 2))
                    pts2.extend(np.float32([cell_kp[m.trainIdx].pt for m in matches]).reshape(-1, 2))
                    grid_matches_prev_idx[j][i] = [m.trainIdx for m in matches]
                    if len(grid_matches_prev_idx[j][i]) == 0:
                        grid_matches_prev_idx[j][i] = None

            # assign cell_kp to grid_prev_kp
            grid_prev_kp[j][i] = cell_kp
            grid_prev_des[j][i] = cell_des

    t2 = perf_counter() - t0 - t1

    if len(pts1) > 0:
        pts1 = np.vstack(pts1)
        pts2 = np.vstack(pts2)
        # discard points that has a distance smaller than min_disparity or larger than max_disparity using np.linalg.norm(np.array(pt1) - np.array(pt2))
        dist = np.linalg.norm(np.array(pts1) - np.array(pts2), axis=1)
        dist_criteria = np.logical_and(dist > min_disparity, dist < max_disparity)
        pts1 = pts1[dist_criteria]
        pts2 = pts2[dist_criteria]

    if len(pts1) > min_n_matches:

        if len(pts1) > subsample_size:
            subsample_idx = np.random.choice(len(pts1), size=subsample_size, replace=False)
            E, submask = cv.findEssentialMat(pts1[subsample_idx], pts2[subsample_idx], focal, pp, method=cv.RANSAC,
                                             prob=0.999999, threshold=1, maxIters=maxIters)
            # Create a full-sized mask, default to 0
            mask = np.zeros(len(pts1), dtype=np.uint8)

            # Set the values of the full-sized mask according to the submask
            mask[subsample_idx] = submask.ravel()
            pts1 = pts1[mask == 1]
            pts2 = pts2[mask == 1]
        else:
            E, mask = cv.findEssentialMat(pts1, pts2, focal, pp, method=cv.RANSAC, prob=0.999999, threshold=1,
                                          maxIters=maxIters)  # Decrease maxIters
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

        t3 = perf_counter() - t0 - t1 - t2

        if len(pts1) > 0:
            _, R, t, _ = cv.recoverPose(E, pts1, pts2)
            t4 = perf_counter() - t0 - t1 - t2 - t3
            # t1 is the ORB instance creation time, t2 is keypoint and matches time, t3 is the essential matrix calculation time, t4 is the recover pose time
            # print the times in ms with 2 decimal places and the name of the time
            print(f"keypoint and matches time: {t2 * 1000:.2f} ms", f"essential matrix calculation time: {t3 * 1000:.2f} ms", f"recover pose time: {t4 * 1000:.2f} ms", "total time: {:.2f} ms".format((t1 + t2 + t3 + t4) * 1000), " FPS: ", int(1/(t1 + t2 + t3 + t4)))

        velocity_dir = R.dot(t.reshape(3, 1))
        velocity_dir = velocity_dir / np.linalg.norm(velocity_dir)
        pixel_coords_hom = np.dot(K, velocity_dir)
        pixel_coords = (pixel_coords_hom[0:2] / pixel_coords_hom[2]).astype(int).flatten()

        if len(pixel_coords) == 4:
            pixel_coords = [pixel_coords[0], pixel_coords[2]]

        # clamp pixel_coords to the image size
        pixel_coords[0] = max(0, min(pixel_coords[0], width))
        pixel_coords[1] = max(0, min(pixel_coords[1], height))

        # Draw the cross at the projected point
        draw_osd(img, width, height, radius=radius, thickness=thickness, cross_size=cross_size, outer_radius=outer_radius)
        # draw a point in the middle of the screen
        img = cv.circle(img, (width // 2, height // 2), radius, color_black, thickness + 1)
        img = cv.circle(img, (width // 2, height // 2), radius - 1, color_white, thickness)

        # draw a circle at the projected point
        if prev_pixel_coords is None:  # first frame
            prev_pixel_coords = np.array(pixel_coords)
        else:
            pixel_coords = (p * np.array(pixel_coords) + (1 - p) * prev_pixel_coords).astype(int)
            prev_pixel_coords = pixel_coords
        img = cv.circle(img, (pixel_coords[0], pixel_coords[1]), marker_size, color_black, thickness + 2)
        img = cv.circle(img, (pixel_coords[0], pixel_coords[1]), marker_size, color_white, thickness)

    if len(pts2)>0:
        # add text to image with len(matches)
        cv.putText(img, str(len(pts2)), (10, 500), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # add text of fps which is 1/(perf_counter() - t0)
        cv.putText(img, str(int(1/(perf_counter() - t0))) + "FPS", (10, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    if is_draw_keypoints:
        if matches is not None:
            if len(pts2) > min_n_matches:
                # draw a line between each keypoint and its maching keypoint from the previous frame
                # Loop over the matches and draw lines between matching points
                for pt1, pt2 in zip(pts1, pts2):
                    pt1 = tuple(np.round(pt1).astype(int))
                    pt2 = tuple(np.round(pt2).astype(int))
                    # Draw line in red color with thickness 1 px
                    cv.line(img, pt1, pt2, (0, 255, 0), 3)
                    # cv.putText(img, str(np.linalg.norm(np.array(pt1) - np.array(pt2)).astype(np.int8)), pt2, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    cv.imshow('ORB Detection Test', img)
    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
cap.close()